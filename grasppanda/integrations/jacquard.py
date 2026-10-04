"""Jacquard observations and explicit planar train/validation protocols."""
import json
import math
from pathlib import Path
import random


DEFAULTS = {'train_fraction': .9, 'split_policy': 'object', 'split_seed': 0, 'iou_threshold': .25}


def options(config):
    return {**DEFAULTS, **config.dataset_options}


def validate_options(config):
    values = options(config)
    if set(values) != set(DEFAULTS):
        raise ValueError('Jacquard dataset_options accepts '+', '.join(DEFAULTS))
    for key, low, high in (('train_fraction', .05, .95), ('iou_threshold', .01, 1.)):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid Jacquard '+key)
    if values['split_policy'] not in ('object', 'ordered_image'):
        raise ValueError('Jacquard split_policy must be object or ordered_image')
    if type(values['split_seed']) is not int or not 0 <= values['split_seed'] < 2**32:
        raise ValueError('Jacquard split_seed must be an integer in [0, 2**32)')
    if config.frame != 0 or config.collision_thresh != 0:
        raise ValueError('Jacquard uses frame: 0 and collision_thresh: 0; rectangles have no 3D collision geometry')
    if config.checkpoint_policy != 'strict':
        raise ValueError('GR-ConvNet retains its native architecture and requires strict checkpoint loading')
    if config.label_root or config.sdf_root:
        raise ValueError('Jacquard labels are beside the images; leave label_root and sdf_root empty')


def inventory(config, split=None):
    root = Path(config.dataset_root)
    if not config.dataset_root or not root.is_dir():
        raise ValueError('Select the Jacquard root containing object folders; use Download starter data')
    files = sorted(root.rglob('*_grasps.txt'))
    if not files:
        raise ValueError('No Jacquard *_grasps.txt files found under the dataset root')
    # Object identity comes from the author filename, independent of archive folders.
    ids = [p.name.split('_', 1)[1].removesuffix('_grasps.txt') for p in files]
    names = [p.name for p in files]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate Jacquard observations found; select one extracted copy of the dataset')
    settings = options(config)
    if settings['split_policy'] == 'object':
        objects = sorted(set(ids))
        if len(objects) < 2:
            raise ValueError('Object-disjoint validation requires at least two Jacquard objects')
        random.Random(settings['split_seed']).shuffle(objects)
        boundary = max(1, min(len(objects)-1, int(len(objects)*settings['train_fraction'])))
        train_objects = set(objects[:boundary])
        train = [p for p, oid in zip(files, ids) if oid in train_objects]
        val = [p for p, oid in zip(files, ids) if oid not in train_objects]
    else:
        if len(files) < 2:
            raise ValueError('Image validation requires at least two Jacquard observations')
        boundary = max(1, min(len(files)-1, int(len(files)*settings['train_fraction'])))
        train, val = files[:boundary], files[boundary:]
    return {'train': train, 'val': val}[split or config.split]


def files_for(label):
    return {'grasps': label, 'depth': label.with_name(label.name.replace('_grasps.txt', '_perfect_depth.tiff')),
            'rgb': label.with_name(label.name.replace('_grasps.txt', '_RGB.png'))}


def selection(config, files):
    if config.action == 'train' and not config.train_batch_limit:
        return list(range(len(files)))
    stop = config.scene+config.frames
    if config.scene < 0 or stop > len(files):
        raise ValueError(f'This {config.split} split has {len(files)} observations; choose a valid first sample and count')
    return list(range(config.scene, stop))


def preflight(config):
    validate_options(config)
    if config.action == 'evaluate' and config.split != 'val':
        raise ValueError('Jacquard evaluation uses split: val')
    files = inventory(config)
    selected = files if config.action == 'evaluate' else [files[i] for i in selection(config, files)]
    if config.action == 'train': selected += inventory(config, 'val')
    for label in selected:
        for role, path in files_for(label).items():
            if not path.is_file():
                raise ValueError(f'Missing Jacquard {role}: {path}')
    if config.checkpoint and not Path(config.checkpoint).is_file():
        raise ValueError('Selected GR-ConvNet checkpoint is missing; download registered weights')
    if config.action in ('infer', 'evaluate') and not config.checkpoint:
        raise ValueError('GR-ConvNet inference/evaluation requires a checkpoint')
    if config.action == 'evaluate':
        manifest = Path(config.prediction_dir)/'manifest.json'
        if not manifest.is_file():
            raise ValueError('Select the predictions folder from a completed Jacquard inference run')
        check_manifest(config, json.loads(manifest.read_text()), files)


def protocol(config):
    return {'dataset': config.dataset, 'method': config.method, 'split': config.split,
            'dataset_options': options(config), 'resolution': 300, 'width_units': 'pixels'}


def check_manifest(config, manifest, files):
    from ..jobs import digest
    if manifest.get('protocol') != protocol(config):
        raise ValueError('Jacquard prediction protocol differs from the selected configuration')
    if manifest.get('checkpoint_sha256') != digest(config.checkpoint):
        raise ValueError('Jacquard predictions used a different checkpoint')
    records = manifest.get('records', [])
    wanted = {p.name for p in files}
    if len(records) != len(wanted) or {r.get('sample') for r in records} != wanted:
        raise ValueError(f'Evaluation requires predictions for every observation in this validation split ({len(wanted)})')
    for row in records:
        name = row.get('prediction', '')
        if Path(name).name != name or not name.endswith('.npz'):
            raise ValueError('Invalid Jacquard prediction filename')
        path = Path(config.prediction_dir)/name
        if not path.resolve().is_relative_to(Path(config.prediction_dir).resolve()) or not path.is_file() or digest(path) != row.get('sha256'):
            raise ValueError('Jacquard prediction file is missing or changed: '+name)
    return records


def frame_cloud(config, scene, frame):
    raise ValueError('Jacquard supplies planar images and rectangles, not calibrated scene point clouds')


def evaluate(config, out):
    from ..methods.grconvnet import evaluate as evaluate_planar
    return evaluate_planar(config, out)
