"""Prepare native GraspClutter6D contact targets beside the dataset."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate_targets(path, require_contacts=False, full=False):
    import h5py
    import numpy as np
    shapes = {'scene_contact_points': (3,), 'scene_grasp_dir': (3,),
              'scene_grasp_app': (3,), 'scene_grasp_rot': (3, 3),
              'scene_grasp_trans': (3,), 'scene_grasp_width': ()}
    try:
        with h5py.File(path, 'r') as data:
            points = data['scene_pc']
            if points.shape != (20000, 3):
                raise ValueError('Expected the native 20000-point training observation')
            contacts = len(data['scene_contact_points'])
            if require_contacts and contacts == 0:
                raise ValueError('This view has no positive contact targets; choose another training view')
            for key, shape in shapes.items():
                value = data[key]
                if value.shape != (contacts, *shape) and not (contacts == 0 and value.shape == (0,)):
                    raise ValueError('Inconsistent contact target shape: '+key)
            if full:
                for key in ('scene_pc', *shapes):
                    if not np.isfinite(data[key][()]).all():
                        raise ValueError('Non-finite contact target: '+key)
            return dict(points=len(points), contacts=contacts)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f'Invalid prepared contact file {path}: {error}') from None


def prepare(root, camera, scenes, frames, seed=0):
    import h5py
    import numpy as np
    import torch
    from grasppanda.config import Experiment
    from grasppanda.integrations.graspclutter6d import official_api, _verify_split, frame_paths, image_id
    from grasppanda.methods.contact_gc6d import prepare as prepare_source
    from grasppanda.jobs import digest

    root = root.expanduser().resolve()
    _verify_split(Experiment(dataset='graspclutter6d', dataset_root=str(root), split='train'))
    if not torch.cuda.is_available():
        raise ValueError('The author contact preprocessing requires a CUDA GPU')
    official_api()
    repo = prepare_source()
    source = repo/'scripts/preprocess_gc6d.py'
    source_sha = digest(source)
    spec = importlib.util.spec_from_file_location('_grasppanda_gc6d_contacts', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.GC6DLoader(str(root), camera=camera, split='train')
    destination = root/'scene_contacts/train'
    destination.mkdir(parents=True, exist_ok=True)
    lock = root/'scene_contacts/.prepare.lock'
    # Serialize writers, including separate CLI/UI processes using this root.
    with lock.open('a') as guard, tempfile.TemporaryDirectory(prefix='.contact-staging-', dir=root) as staging:
        fcntl.flock(guard, fcntl.LOCK_EX)
        options = SimpleNamespace(dataset_root=staging, split='train', camera=camera,
                                  num_points=20000, vis=False)
        for scene in scenes:
            collision = None
            labels = None
            grasp_hashes = {}
            for frame in frames:
                target = destination/f'{scene:06d}_{image_id(camera, frame):06d}.h5'
                if target.is_file():
                    details = validate_targets(target, full=True)
                    with h5py.File(target, 'r') as data:
                        if data.attrs.get('grasppanda_dataset') == 'graspclutter6d':
                            for name, expected in (('camera',camera),('scene',scene),('frame',frame),
                                                   ('seed',seed),('preprocessor_sha256',source_sha)):
                                if data.attrs.get(name) != expected:
                                    raise ValueError(f'Existing contact targets have a different {name}: {target}. Keep their original settings or move the file before regenerating.')
                    print(json.dumps(dict(scene=scene, frame=frame, state='reused', **details)), flush=True)
                    continue
                inputs = frame_paths(root, scene, camera, frame)
                inputs['poses'] = root/'scenes'/f'{scene:06d}'/'scene_gt.json'
                inputs['collision'] = root/'collision_label'/f'{scene:06d}.npz'
                for role, path in inputs.items():
                    if not path.is_file():
                        raise ValueError(f'Missing {role} input: {path}')
                if labels is None:
                    objects, _ = loader.loadSceneObjectList(scene, 0, camera)
                    for obj in objects:
                        path = root/'grasp_label'/f'obj_{obj:06d}_labels.npz'
                        if not path.is_file():
                            raise ValueError(f'Missing object grasp labels: {path}')
                        grasp_hashes[str(path.relative_to(root))] = digest(path)
                    labels = module.load_scene_grasp_labels(loader, scene, camera)
                    collision = loader.loadCollisionLabels(scene)
                frame_seed = int(np.random.SeedSequence([seed, scene, image_id(camera, frame)]).generate_state(1)[0])
                np.random.seed(frame_seed)
                torch.manual_seed(frame_seed)
                module.process_scene(options, loader, scene, frame, labels, collision)
                generated = Path(staging)/'scene_contacts/train'/target.name
                details = validate_targets(generated, full=True)
                with h5py.File(generated, 'a') as data:
                    data.attrs['grasppanda_dataset'] = 'graspclutter6d'
                    data.attrs['camera'] = camera
                    data.attrs['scene'] = scene
                    data.attrs['frame'] = frame
                    data.attrs['seed'] = seed
                    data.attrs['preprocessor_sha256'] = source_sha
                    data.attrs['input_sha256'] = json.dumps({role: digest(path) for role, path in inputs.items()})
                    data.attrs['grasp_label_sha256'] = json.dumps(grasp_hashes)
                generated.replace(target)
                print(json.dumps(dict(scene=scene, frame=frame, state='prepared', **details)), flush=True)


def main():
    from grasppanda.datasets import get_dataset
    from grasppanda.config import default_dataset
    spec = get_dataset('graspclutter6d')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=default_dataset('graspclutter6d'))
    parser.add_argument('--camera', choices=spec.cameras, default='realsense-d435')
    parser.add_argument('--scenes', type=int, nargs='+', help='Training scene IDs; omitted means the full published training split')
    parser.add_argument('--frames', type=int, nargs='+', help='Ordinal views 0–12; omitted means all views')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    scenes = sorted(set(args.scenes if args.scenes is not None else spec.scene_ids('train')))
    frames = sorted(set(args.frames if args.frames is not None else range(spec.frames_per_scene)))
    if not args.dataset_root:
        parser.error('Set --dataset-root or GRASPPANDA_GRASPCLUTTER6D_ROOT')
    if not set(scenes).issubset(spec.scene_ids('train')):
        parser.error('Use scene IDs from the published grasp training split')
    if not set(frames).issubset(range(spec.frames_per_scene)) or args.seed < 0:
        parser.error('Frames must be ordinal views 0–12 and seed must be nonnegative')
    try:
        prepare(Path(args.dataset_root), args.camera, scenes, frames, args.seed)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
