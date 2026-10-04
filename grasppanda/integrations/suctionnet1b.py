"""Suction-specific observations and training labels on shared GraspNet scenes."""
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_options(config):
    if config.dataset_options:
        raise ValueError('SuctionNet currently retains the native label and decoding parameters')
    if config.workspace != 'native_image' or config.collision_thresh != 0:
        raise ValueError('SuctionNet requires native_image and collision_thresh: 0; parallel-jaw collision filtering does not apply')
    if config.checkpoint_policy != 'strict' or config.sdf_root:
        raise ValueError('SuctionNet requires strict checkpoint loading and an empty sdf_root')
    if config.eval_batch_limit:
        raise ValueError('SuctionNet training has no held-out validation loop; native benchmark evaluation remains upstream')
    if config.action in ('train', 'train_short') and config.batch_size < 2:
        raise ValueError('Native SuctionNet ASPP batch normalization requires batch_size >= 2')


def selection(config):
    from ..datasets import get_dataset
    spec = get_dataset(config.dataset)
    if config.action == 'train' and not config.train_batch_limit:
        return [(scene, frame) for scene in spec.scene_ids('train') for frame in range(256)]
    return list(spec.frame_keys(config.split, config.scene, config.frame, config.frames))


def files_for(config, scene, frame):
    base = Path(config.dataset_root)/f'scenes/scene_{scene:04d}'/config.camera
    return {role: base/f'{role}/{frame:04d}.{ext}' for role, ext in
            (('rgb', 'png'), ('depth', 'png'), ('meta', 'mat'), ('annotations', 'xml'))}


def label_inputs(config, scene, frame):
    root = Path(config.dataset_root)
    files = files_for(config, scene, frame)
    labels = [files['annotations'], files_for(config, scene, 0)['meta'],
              root/f'suction_collision_label/{scene:04d}_collision.npz']
    for entry in ET.parse(files['annotations']).getroot().findall('obj/obj_id'):
        index = int(entry.text)
        labels += [root/f'models/{index:03d}/nontextured.ply', root/f'seal_label/{index:03d}_seal.npz']
    return labels


def cache_root(config):
    return Path(config.label_root) if config.label_root else Path(config.dataset_root)/'.grasppanda/suction-v1'


def preflight(config):
    validate_options(config)
    if not config.dataset_root or not Path(config.dataset_root).is_dir():
        raise ValueError('Select a SuctionNet root containing scenes/; existing GraspNet images may be linked there')
    chosen = selection(config)
    training = config.action in ('train', 'train_short')
    if training and len(chosen) < 2:
        raise ValueError('SuctionNet training requires at least two selected frames; set frames >= 2')
    if training and len(chosen) < config.batch_size:
        raise ValueError('SuctionNet drops incomplete training batches; select at least batch_size frames')
    checked_scenes = set()
    for scene, frame in chosen:
        files = files_for(config, scene, frame)
        for role in ('rgb', 'depth', 'meta', 'annotations') if training else ('rgb', 'depth', 'meta'):
            if not files[role].is_file():raise ValueError(f'Missing SuctionNet {role}: {files[role]}')
        if training and scene not in checked_scenes:
            for path in label_inputs(config, scene, frame):
                if not path.is_file():
                    raise ValueError('Missing SuctionNet training input: '+str(path)+'. See Guide → Datasets.')
            checked_scenes.add(scene)
    if config.checkpoint and not Path(config.checkpoint).is_file():
        raise ValueError('SuctionNet checkpoint missing; use Download registered weights')
    if config.action == 'infer' and not config.checkpoint:
        raise ValueError('SuctionNet inference requires a trained checkpoint')


def protocol(config):
    return {'dataset':config.dataset, 'method':config.method, 'camera':config.camera,
            'split':config.split, 'input':'BGR / 255 and depth metres clipped to [0, 1]',
            'output_columns':['score', 'nx', 'ny', 'nz', 'x', 'y', 'z'],
            'coordinate_frame':'camera', 'translation_units':'metres',
            'label_projection':'author source, Gaussian sigma 4; training scene 51 excluded'}


def frame_cloud(config, scene, frame):
    raise ValueError('SuctionNet uses native image inference and suction normals, not parallel-jaw grasp tensors')


def evaluate(config, out):
    raise ValueError('SuctionNet AP requires the separate author suction evaluator and dense point-cloud release; see Guide → Datasets')
