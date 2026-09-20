"""GraspClutter6D camera geometry, published splits and official evaluation."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

from ..datasets import get_dataset
from ..jobs import digest


CAMERA_OFFSETS = {'realsense-d415': 1, 'realsense-d435': 2, 'azure-kinect': 3, 'zivid': 4}
DEPTH_FACTORS = {'realsense-d415': 1000., 'realsense-d435': 1000., 'azure-kinect': 10000., 'zivid': 10000.}


def image_id(camera, frame):
    if camera not in CAMERA_OFFSETS:
        raise ValueError(f'Unknown GraspClutter6D camera: {camera}')
    if type(frame) is not int or not 0 <= frame < 13:
        raise ValueError('GraspClutter6D frame must be an ordinal in [0, 12]')
    return frame * 4 + CAMERA_OFFSETS[camera]


def prediction_path(scene, camera, frame):
    return Path(f'{scene:06d}') / camera / f'{image_id(camera, frame):06d}.npy'


def frame_paths(root, scene, camera, frame):
    folder = Path(root) / 'scenes' / f'{scene:06d}'
    name = f'{image_id(camera, frame):06d}.png'
    return {'depth': folder/'depth'/name, 'rgb': folder/'rgb'/name,
            'label': folder/'label'/name, 'camera': folder/'scene_camera.json'}


def _verify_split(config):
    name = Path(config.dataset_root)/'split_info'/f'grasp_{config.split}_scene_ids.json'
    if not name.is_file():
        raise ValueError(f'Missing published grasp split: {name}. Extract split_info.7z into the dataset root.')
    try:
        actual = tuple(int(value) for value in json.loads(name.read_text()))
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid GraspClutter6D split file: {name}') from error
    expected = get_dataset(config.dataset).scene_ids(config.split)
    if actual != expected:
        raise ValueError('The scene list differs from the pinned GraspClutter6D grasp split; YCB pose splits are a different protocol')


def preflight(config):
    root = Path(config.dataset_root)
    if not config.dataset_root or not (root/'scenes').is_dir():
        raise ValueError('Select the GraspClutter6D root containing scenes/ and split_info/')
    if config.workspace not in ('official_gt_workspace', 'depth_only'):
        raise ValueError('GraspClutter6D point inference requires official_gt_workspace or depth_only')
    _verify_split(config)
    spec = get_dataset(config.dataset)
    if config.action == 'infer':
        if not config.checkpoint or not Path(config.checkpoint).is_file():
            raise ValueError('Select a compatible local checkpoint for this method; use Download registered weights')
        for scene, frame in spec.frame_keys(config.split, config.scene, config.frame, config.frames):
            paths = frame_paths(root, scene, config.camera, frame)
            for role, path in paths.items():
                if role == 'label' and config.workspace == 'depth_only':
                    continue
                if not path.is_file():
                    raise ValueError(f'Missing GraspClutter6D {role} input: {path}')
    elif config.action in ('train', 'train_short') and config.method == 'contact_graspnet_gc6d':
        from ..methods.contact_gc6d_training import target_paths
        from ..runtime.prepare_contacts import validate_targets
        paths = target_paths(config)
        usable = 0
        for path in paths:
            usable += validate_targets(path, require_contacts=config.action == 'train_short')['contacts'] > 0
        if not usable:
            raise ValueError('The selected training inputs contain no positive contact labels')
        if config.checkpoint and not Path(config.checkpoint).is_file():
            raise ValueError('Selected training checkpoint is missing; download it or clear the path to train from scratch')
    elif config.action == 'evaluate':
        if config.split != 'test':
            raise ValueError('Official GraspClutter6D AP uses the published test split')
        directory = Path(config.prediction_dir)
        manifest_path = directory/'manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Prediction provenance missing: manifest.json is required')
        manifest = json.loads(manifest_path.read_text())
        for key in ('dataset', 'method', 'modules', 'camera', 'split', 'workspace',
                    'collision_thresh', 'voxel_size', 'num_points'):
            if manifest.get('config', {}).get(key) != getattr(config, key):
                raise ValueError(f'Prediction protocol mismatch: {key}')
        if manifest.get('config', {}).get('refinement'):
            raise ValueError('GraspClutter6D refinement evaluation has not been adapted')
        if not config.checkpoint or not Path(config.checkpoint).is_file() or manifest.get('checkpoint_sha256') != digest(config.checkpoint):
            raise ValueError('Selected checkpoint differs from the prediction manifest')
        expected = {str(prediction_path(scene, config.camera, frame))
                    for scene in spec.scene_ids('test') for frame in range(spec.frames_per_scene)}
        if set(manifest.get('files', {})) != expected:
            raise ValueError('Official AP requires predictions for all 235 test scenes and 13 views of the selected camera')
        for relative in sorted(expected):
            file = directory/relative
            if not file.is_file() or digest(file) != manifest['files'][relative]:
                raise ValueError(f'Missing or changed prediction: {relative}')
        for folder in ('models_m', 'dex_models'):
            if not (root/folder).is_dir():
                raise ValueError(f'Official evaluation also needs {folder}/; see the dataset downloads guide')
        official_api()
    else:
        raise ValueError(f'No GraspClutter6D {config.action} adapter for {config.method}')


def frame_cloud(config, scene, frame):
    import numpy as np
    from PIL import Image
    paths = frame_paths(config.dataset_root, scene, config.camera, frame)
    depth = np.asarray(Image.open(paths['depth']))
    if depth.ndim != 2:
        raise ValueError('GraspClutter6D depth must be a single-channel image')
    with Image.open(paths['rgb']) as rgb:
        if rgb.size != (depth.shape[1], depth.shape[0]):
            raise ValueError('RGB and depth are not aligned at the same resolution')
    cameras = json.loads(paths['camera'].read_text())
    calibration = cameras.get(str(image_id(config.camera, frame)))
    if calibration is None:
        raise ValueError('Camera calibration is missing for the selected camera/view')
    intr = np.asarray(calibration['cam_K'], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(intr).all() or intr[0, 0] <= 0 or intr[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    # Match the official API: RealSense stores mm, Azure/Zivid store 0.1 mm.
    z = depth.astype(np.float64) / DEPTH_FACTORS[config.camera]
    y, x = np.indices(depth.shape)
    cloud = np.stack(((x-intr[0, 2])*z/intr[0, 0],
                      (y-intr[1, 2])*z/intr[1, 1], z), axis=-1)
    evidence = {f'{key}_sha256': digest(paths[key]) for key in ('depth', 'rgb', 'camera')}
    evidence['image_id'] = image_id(config.camera, frame)
    if config.workspace == 'official_gt_workspace':
        labels = np.asarray(Image.open(paths['label']))
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        if labels.shape != depth.shape:
            raise ValueError('Instance labels are not aligned with the depth image')
        columns, rows = np.any(labels > 0, axis=0), np.any(labels > 0, axis=1)
        if not columns.any() or not rows.any():
            raise ValueError('The selected view has no foreground for the official instance-mask workspace')
        width, height = depth.shape[1], depth.shape[0]
        x1, y1 = int(columns.argmax()), int(rows.argmax())
        x2, y2 = width-int(columns[::-1].argmax()), height-int(rows[::-1].argmax())
        margin_x, margin_y = int(width*.1), int(height*.1)
        cloud = cloud[max(0, y1-margin_y):min(height, y2+margin_y),
                      max(0, x1-margin_x):min(width, x2+margin_x)]
        evidence['label_sha256'] = digest(paths['label'])
    raw = cloud[(cloud[..., 2] > 0) & np.isfinite(cloud).all(axis=-1)].astype(np.float32)
    if len(raw) == 0:
        raise ValueError('The selected GraspClutter6D view has no valid depth points')
    if config.method == 'contact_graspnet_gc6d':
        indices = np.random.choice(len(raw), config.num_points, replace=len(raw) < config.num_points)
    elif len(raw) >= config.num_points:
        indices = np.random.choice(len(raw), config.num_points, replace=False)
    else:
        indices = np.concatenate((np.arange(len(raw)), np.random.choice(len(raw), config.num_points-len(raw), replace=True)))
    return raw, raw[indices], intr, paths['rgb'], evidence


def official_api():
    from ..config import ROOT
    path = ROOT/'upstream/infrastructure/graspclutter6d_api'
    pins = json.loads((ROOT/'grasppanda/resources/upstreams.lock.json').read_text())
    pin = next((entry for entry in pins if entry['id'] == 'graspclutter6d_api'), None)
    if pin is None or not (path/'.git').exists():
        raise ValueError('The pinned GraspClutter6D API is missing; run ./panda fetch')
    sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != pin['pinned_commit']:
        raise ValueError('GraspClutter6D API differs from the registered source revision')
    changes = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if changes:
        raise ValueError('GraspClutter6D API has modified tracked files; restore the pinned implementation')
    sys.path.insert(0, str(path))
    module = importlib.import_module('graspclutter6dAPI')
    if not Path(module.__file__).resolve().is_relative_to(path.resolve()):
        raise ValueError('A different GraspClutter6D API is already imported; start a fresh worker')
    return module


def evaluate(config, out):
    import numpy as np
    evaluator = official_api().GraspClutter6DEval(root=config.dataset_root, camera=config.camera, split='test')
    result, ap = evaluator.eval_all(dump_folder=config.prediction_dir, proc=2)
    if not np.isfinite(result).all() or not np.isfinite(ap).all():
        raise ValueError('The official evaluator returned non-finite results')
    np.save(out/'evaluation.npy', result)
    return {'stage': 'official_evaluation', 'dataset': config.dataset, 'split': config.split,
            'camera': config.camera, 'method': config.method, 'ap': np.asarray(ap).tolist(),
            'ap_names': ['AP', 'AP0.4', 'AP0.8'], 'prediction_dir': config.prediction_dir,
            'prediction_manifest_sha256': digest(Path(config.prediction_dir)/'manifest.json')}
