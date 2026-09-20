"""Rendered depth observations and native LEAP-hand supervision for DexGraspNet 2.0."""
from pathlib import Path

METHODS = ('dexgraspnet2', 'dexgraspnet2_isa', 'dexgraspnet2_cvae')
JOINT_NAMES = ('j12', 'j13', 'j14', 'j15', 'j1', 'j0', 'j2', 'j3',
               'j9', 'j8', 'j10', 'j11', 'j5', 'j4', 'j6', 'j7')


def frame_paths(root, scene, camera, frame):
    base = Path(root)/'scenes'/f'scene_{scene:04d}'/camera
    return dict(depth=base/'depth_gt'/f'{frame:04d}.png',
                label=base/'label_gt'/f'{frame:04d}.png',
                calibration=base/'meta'/f'{frame:04d}.mat',
                poses=base/'camera_poses.npy', table=base/'cam0_wrt_table.npy')


def label_paths(root, scene, camera, frame):
    root = Path(root)
    graspness = root/'dex_graspness_new'/f'scene_{scene:04d}'/camera/f'{frame:04d}.npy'
    grasps = sorted((root/'dex_grasps_new'/f'scene_{scene:04d}'/'leap_hand').glob('*.npz'))
    if not graspness.is_file() or not grasps:
        raise ValueError(f'Missing native graspness or LEAP-hand targets for scene {scene}, view {frame}. See Guide → Datasets.')
    return graspness, grasps


def pairs(config):
    from ..datasets import get_dataset
    return get_dataset(config.dataset).frame_keys(config.split, config.scene, config.frame, config.frames)


def preflight(config):
    if config.method not in METHODS or config.action not in ('infer', 'train', 'train_short'):
        raise ValueError('DexGraspNet 2.0 supports native hand prediction and supervised training; simulation evaluation is separate.')
    if config.workspace != 'official_gt_workspace':
        raise ValueError('This DexGraspNet adapter uses the native rendered-depth and instance-workspace protocol.')
    if config.collision_thresh != 0:
        raise ValueError('Set collision_thresh: 0. A parallel-gripper collision threshold does not define LEAP-hand collision checking.')
    if config.label_root or config.eval_batch_limit or config.loss or config.augmentation or config.optimizer or config.scheduler:
        raise ValueError('DexGraspNet uses native targets under dataset_root, native losses, rotation augmentation, Adam and the 50000-update cosine schedule.')
    if config.checkpoint and not Path(config.checkpoint).is_file():
        raise ValueError('Selected DexGraspNet checkpoint is missing. Download registered weights first.')
    if config.action == 'infer' and not config.checkpoint:
        raise ValueError('DexGraspNet prediction requires a compatible trained checkpoint.')
    selection = pairs(config)
    first = next(selection)
    # Inspect the first selected observation before allocating the model. Later
    # missing views raise a precise error instead of the upstream loader's recursion.
    observation(config, *first)
    if config.action in ('train', 'train_short'):
        if config.frames < config.batch_size:
            raise ValueError('Select at least batch_size training views, or reduce batch_size.')
        label_paths(config.dataset_root, first[0], config.camera, first[1])


def observation(config, scene, frame, rng=None):
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat
    from ..jobs import digest
    paths = frame_paths(config.dataset_root, scene, config.camera, frame)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f'Missing DexGraspNet observation: {path}. Extract the scene archives before running.')
    depth = np.asarray(Image.open(paths['depth']))
    labels = np.asarray(Image.open(paths['label']))
    meta = loadmat(paths['calibration'])
    intrinsics = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    factor = float(np.asarray(meta['factor_depth']).reshape(-1)[0])
    poses = np.load(paths['poses'], allow_pickle=False)
    table = np.load(paths['table'], allow_pickle=False)
    if (depth.ndim != 2 or labels.shape != depth.shape or intrinsics.shape != (3,3)
            or not np.isfinite(intrinsics).all() or min(intrinsics[0,0], intrinsics[1,1], factor) <= 0
            or not np.isfinite(factor) or poses.shape != (256,4,4) or table.shape != (4,4)
            or not np.isfinite(poses).all() or not np.isfinite(table).all()):
        raise ValueError('Invalid rendered depth, instance labels or camera calibration.')
    y, x = np.indices(depth.shape)
    z = depth/factor
    cloud = np.stack(((x-intrinsics[0,2])*z/intrinsics[0,0],
                      (y-intrinsics[1,2])*z/intrinsics[1,1], z), axis=-1)
    # Author get_workspace_mask: table-frame foreground bounds plus 2 cm.
    transform = table@poses[frame]
    aligned = cloud@transform[:3,:3].T+transform[:3,3]
    foreground = aligned[labels>0]
    if not len(foreground):
        raise ValueError('The selected observation contains no foreground instance pixels.')
    mask = (depth>0) & np.all((aligned>foreground.min(0)-.02) & (aligned<foreground.max(0)+.02), axis=-1)
    raw = cloud[mask]
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('The native workspace contains no finite depth points.')
    rng = rng if rng is not None else np.random.RandomState(config.seed+frame+scene*256)
    indices = rng.choice(len(raw), config.num_points, replace=True)
    return dict(raw=raw, points=raw[indices].astype(np.float32), labels=labels[mask][indices].astype(np.int64),
                indices=indices, mask=mask, depth=depth, intrinsics=intrinsics, pose=poses[frame], table_pose=transform,
                evidence={str(path.relative_to(config.dataset_root)):digest(path) for path in paths.values()})


def frame_cloud(config, scene, frame):
    raise ValueError('DexGraspNet observations require the native hand adapter and its explicit output contract.')


def evaluate(config, out):
    raise ValueError('DexGraspNet reports Isaac Gym grasp success, not parallel-gripper AP. See the native simulation workflow in Guide → Datasets.')


class TrainingViews:
    """Finite selected-view epochs, retaining the author's target construction."""
    def __init__(self, config):
        from ..datasets import get_dataset
        self.config, self.epoch = config, 0
        spec = get_dataset(config.dataset)
        self.scenes = spec.scene_ids(config.split)
        self.start = self.scenes.index(config.scene)*256+config.frame

    def __len__(self):
        return self.config.frames

    def __getitem__(self, index):
        import numpy as np
        from ..jobs import digest
        scene_index, frame = divmod(self.start+index, 256)
        scene = self.scenes[scene_index]
        seed = (self.config.seed+self.epoch*1000003+index*97) % (2**32)
        rng = np.random.RandomState(seed)
        data = observation(self.config, scene, frame, rng)
        graspness_path, grasp_files = label_paths(self.config.dataset_root, scene, self.config.camera, frame)
        graspness = np.load(graspness_path, allow_pickle=False).reshape(-1)
        if len(graspness) != len(data['raw']) or not np.isfinite(graspness).all() or (graspness<0).any():
            raise ValueError('Graspness targets do not align with the native rendered-depth workspace.')
        graspness = np.log(graspness[data['indices']]+1e-3).astype(np.float32)
        evidence = dict(data['evidence'])
        evidence[str(graspness_path.relative_to(self.config.dataset_root))] = digest(graspness_path)
        objects = []
        for path in grasp_files:
            with np.load(path, allow_pickle=False) as archive:
                values = {k: archive[k] for k in ('point','rotation','translation',*JOINT_NAMES)}
            size = len(values['point'])
            if not size or any(len(v)!=size or not np.isfinite(v).all() for v in values.values()):
                raise ValueError('Native dexterous grasp rows are empty, non-finite or misaligned: '+path.name)
            objects.append(values)
            evidence[str(path.relative_to(self.config.dataset_root))] = digest(path)
        # Equal object sampling, 128 proposals, native 6 mm matching and 64
        # supervised grasp seeds. Retry only the selected view, with a bound.
        cloud, camera = data['points'], data['pose']
        for attempt in range(16):
            assignments = rng.randint(len(objects), size=128)
            selected = []
            for oid, obj in enumerate(objects):
                ids = rng.choice(len(obj['point']), int((assignments==oid).sum()), replace=True)
                selected.append({key:value[ids] for key,value in obj.items()})
            order = rng.permutation(128)
            selected = {key:np.concatenate([obj[key] for obj in selected])[order] for key in selected[0]}
            points = (selected['point']-camera[:3,3])@camera[:3,:3]
            available, centers = [], []
            for row, point in enumerate(points):
                distances = np.linalg.norm(cloud-point, axis=1)
                nearest = int(distances.argmin())
                if distances[nearest] <= .006:
                    available.append(row); centers.append(nearest)
                    if len(available) == 64: break
            if available: break
        if not available:
            raise ValueError(f'No labeled hand contact lies within 6 mm of sampled depth in scene {scene}, view {frame}; choose another view or more points.')
        chosen = rng.choice(len(available), 64, replace=True)
        rows = np.asarray(available)[chosen]
        rotation = np.einsum('ji,njk->nik', camera[:3,:3], selected['rotation'][rows])
        translation = (selected['translation'][rows]-camera[:3,3])@camera[:3,:3]
        qpos = np.stack([selected[j][rows] for j in JOINT_NAMES], axis=-1)
        theta = rng.rand()*2*np.pi
        augmentation = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]],dtype=np.float32)
        cloud = cloud@augmentation.T
        native = dict(point_clouds=cloud.astype(np.float32), coors=cloud.astype(np.float32)/self.config.voxel_size,
            feats=np.ones_like(cloud,dtype=np.float32), seg=data['labels'], objectness=(data['labels']>0).astype(np.int64),
            graspness=graspness, rot=np.einsum('ij,njk->nik',augmentation,rotation).astype(np.float32),
            trans=(translation@augmentation.T).astype(np.float32), qpos=qpos.astype(np.float32),
            centers=np.asarray(centers,dtype=np.float32)[chosen], has_graspness=np.array([1]))
        return native, evidence


def collate(samples):
    from src.utils.dataset import minkowski_collate_fn
    return minkowski_collate_fn([value for value,_ in samples]), [evidence for _,evidence in samples]
