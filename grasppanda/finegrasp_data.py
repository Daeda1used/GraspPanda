"""Native FineGrasp samples with reproducible, locally cached derived inputs."""
import json
from pathlib import Path


def instance_graspness(values, instances):
    """Paper equations 1–2; constant objects and background receive zero."""
    import numpy as np
    values = np.asarray(values).reshape(-1)
    instances = np.asarray(instances).reshape(-1)
    if len(values) != len(instances) or not np.isfinite(values).all():
        raise ValueError('Graspness rows must match the native workspace point order')
    normalized = np.zeros_like(values, dtype=np.float64)
    for instance in np.unique(instances):
        if instance == 0: continue
        selected = instances == instance
        group = values[selected]
        low, high = group.min(), group.max()
        if high > low: normalized[selected] = (group - low) / (high - low)
    low, high = normalized.min(), normalized.max()
    if high > low: normalized = (normalized - low) / (high - low)
    return normalized.astype(np.float32)[:, None]


def normals(points):
    from .finegrasp import native_module
    native_module()
    from robo_orchard_lab.models.finegrasp.processor import FineGraspProcessor
    # The native constructor resets global RNGs; this stateless method does not.
    return FineGraspProcessor.get_normal(None, points)


def save_array(path, array):
    import os
    import tempfile
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, array)
    try: os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


class FineGraspDataset:
    def __init__(self, config, cache, augment=True):
        from types import SimpleNamespace
        from .finegrasp import native_module
        native_module()
        from robo_orchard_lab.dataset.graspnet1b import EconomicGraspNet1BDataset
        self.config, self.cache, self.augment = config, Path(cache), augment
        self.native = EconomicGraspNet1BDataset(SimpleNamespace(
            data_root=config.dataset_root, camera=config.camera, split='train',
            voxel_size=config.voxel_size, num_sample_points=config.num_points,
            remove_outlier=True, remove_invisible=True, use_new_graspness=True,
            augment=False, load_label=True))

    def __len__(self): return len(self.native)

    def prepare(self, index):
        from .finegrasp import native_module
        native_module()
        import fcntl
        import numpy as np
        import scipy.io
        import torch
        from PIL import Image
        from .jobs import digest
        from robo_orchard_lab.utils.geometry import depth_to_range_image
        scene, frame = divmod(index, 256)
        root = Path(self.config.dataset_root)
        folder = root/'scenes'/f'scene_{scene:04d}'/self.config.camera
        original_normal = folder/'normal'/f'{frame:04d}.npy'
        original_graspness = root/'instance_norm_graspness'/f'scene_{scene:04d}'/self.config.camera/f'{frame:04d}.npy'
        self.native.normalpath[index] = str(original_normal)
        self.native.graspnesspath[index] = str(original_graspness)
        if original_normal.is_file() and original_graspness.is_file(): return
        cache = self.cache/f'scene_{scene:04d}'/self.config.camera/f'{frame:04d}'
        cache.mkdir(parents=True, exist_ok=True)
        with (cache/'.prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            paths = {'depth': folder/'depth'/f'{frame:04d}.png', 'segmentation': folder/'label'/f'{frame:04d}.png',
                     'metadata': folder/'meta'/f'{frame:04d}.mat', 'camera_poses': folder/'camera_poses.npy',
                     'table_pose': folder/'cam0_wrt_table.npy'}
            if not original_graspness.is_file():
                paths['scene_graspness'] = root/'graspness'/f'scene_{scene:04d}'/self.config.camera/f'{frame:04d}.npy'
            inputs = {name: digest(path) for name, path in paths.items()}
            inputs['adapter'] = digest(__file__)
            import inspect
            import open3d
            from robo_orchard_lab.models.finegrasp.processor import FineGraspProcessor
            inputs['dataset_reader'] = digest(inspect.getfile(type(self.native)))
            inputs['normal_estimator'] = digest(inspect.getfile(FineGraspProcessor))
            inputs['open3d'] = open3d.__version__
            manifest = cache/'prepared.json'
            expected = {name: cache/(name+'.npy') for name, original in
                        (('normal', original_normal), ('graspness', original_graspness)) if not original.is_file()}
            if manifest.is_file():
                try: previous = json.loads(manifest.read_text())
                except (ValueError, OSError): previous = {}
                valid = previous.get('inputs') == inputs and all(path.is_file() and previous.get('outputs', {}).get(name) == digest(path) for name, path in expected.items())
            else: valid = False
            if not valid:
                depth = np.array(Image.open(paths['depth']))
                segmentation = np.array(Image.open(paths['segmentation']))
                meta = scipy.io.loadmat(paths['metadata'])
                cloud = depth_to_range_image(depth, meta['intrinsic_matrix'], depth_scale=meta['factor_depth'])
                transform = np.load(paths['table_pose']) @ np.load(paths['camera_poses'])[frame]
                workspace = self.native.get_workspace_mask(torch.from_numpy(cloud), torch.from_numpy(segmentation),
                    trans=transform, organized=True, outlier=.02)
                mask = (depth > 0) & np.asarray(workspace)
                if not mask.any(): raise ValueError('FineGrasp frame has no valid workspace points')
                if 'graspness' in expected:
                    values = np.load(paths['scene_graspness'])
                    save_array(expected['graspness'], instance_graspness(values, segmentation[mask]))
                if 'normal' in expected:
                    full = np.zeros_like(cloud, dtype=np.float32)
                    full[mask] = normals(cloud[mask]) * 255.
                    # Native dataset divides its signed floating-point normal map by 255.
                    save_array(expected['normal'], full)
                temporary = cache/'prepared.json.tmp'
                temporary.write_text(json.dumps(dict(inputs=inputs, outputs={name: digest(path) for name, path in expected.items()},
                    normal_estimator='Native Open3D hybrid search: radius 0.1 m, max_nn 30; full workspace cloud',
                    graspness='Per-foreground-instance min-max, then scene min-max; constant instances/background zero'), indent=2)+'\n')
                temporary.replace(manifest)
            for name, path in expected.items():
                getattr(self.native, 'normalpath' if name == 'normal' else 'graspnesspath')[index] = str(path)

    def __getitem__(self, index):
        import numpy as np
        from .training_options import sample_points, transform_points
        self.prepare(index)
        sample = self.native[index]
        options = self.config.augmentation
        mode = options.get('mode', 'custom') if options else 'native'
        if self.augment and mode != 'none':
            count = len(sample['point_clouds'])
            if mode == 'native':
                combined = np.concatenate([sample['point_clouds'], sample['cloud_normal']])
                combined, poses = self.native.augment_data(combined, sample['object_poses_list'])
                sample['point_clouds'], sample['cloud_normal'] = combined[:count].astype(np.float32), combined[count:].astype(np.float32)
                sample['object_poses_list'] = poses
            else:
                poses = sample['object_poses_list'] + [np.eye(4, dtype=np.float32)[:3]]
                points, transformed = transform_points(sample['point_clouds'], poses, options)
                sample['point_clouds'], sample['object_poses_list'] = points, transformed[:-1]
                sample['cloud_normal'] = (sample['cloud_normal'] @ transformed[-1][:3, :3].T).astype(np.float32)
                # Use the same sample map for every observation feature and label.
                sample = sample_points(sample, self.config)
                if options.get('depth_noise_std', 0) or options.get('jitter_std', 0):
                    sample['cloud_normal'] = normals(sample['point_clouds']).astype(np.float32)
        sample['coordinates_for_voxel'] = sample['point_clouds'] / self.config.voxel_size
        if not np.isfinite(sample['cloud_normal']).all(): raise ValueError('FineGrasp normals are not finite')
        return sample
