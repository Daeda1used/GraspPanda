"""Dataset registry with explicit executable providers and metadata."""
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    cameras: tuple[str,...]
    splits: dict[str,tuple[int,int]]
    frames_per_scene: int
    modalities: tuple[str,...]
    download_url: str
    scene_splits: dict[str,tuple[int,...]] = field(default_factory=dict)
    method_actions: dict[str,tuple[str,...]] | None = None
    protocol_note: str = ''
    default_method: str = 'graspnet_baseline'
    scene_label: str = 'First scene'
    frame_label: str = 'First frame'
    runner: str | None = None
    hidden_controls: tuple[str,...] = ()

    @property
    def inference_split(self):
        return next((name for name in ('test_seen', 'test', 'val', 'valid') if name in self.splits), next(iter(self.splits)))

    def scene_ids(self, split):
        if split not in self.splits:
            raise ValueError(f'Unknown {self.title} split: {split}')
        if split in self.scene_splits:
            return self.scene_splits[split]
        # Large object datasets need bounds and indexing, not millions of allocated IDs.
        return range(*self.splits[split])

    def frame_keys(self, split, scene, frame, count):
        """Walk the declared split, including non-contiguous scene indices."""
        scenes = self.scene_ids(split)
        if scene not in scenes:
            raise ValueError(f'Scene {scene} is not in the {self.title} {split} split')
        if type(frame) is not int or not 0 <= frame < self.frames_per_scene:
            raise ValueError(f'Frame must be in [0, {self.frames_per_scene - 1}]')
        start = scenes.index(scene) * self.frames_per_scene + frame
        if type(count) is not int or count < 1 or start + count > len(scenes) * self.frames_per_scene:
            raise ValueError('Requested frame range crosses the selected split')
        for index in range(start, start + count):
            scene_index, frame_index = divmod(index, self.frames_per_scene)
            yield scenes[scene_index], frame_index

_DATASETS={}
_PROVIDERS={}


def register_dataset(spec,provider=None):
    if spec.key in _DATASETS:raise ValueError(f'Dataset already registered: {spec.key}')
    if not spec.key or spec.frames_per_scene<=0:raise ValueError('Invalid dataset specification')
    for split, scenes in spec.scene_splits.items():
        if split not in spec.splits or not scenes or tuple(sorted(set(scenes))) != scenes:
            raise ValueError('Explicit scene splits must be nonempty, sorted and unique')
        low, high = spec.splits[split]
        if any(type(scene) is not int or not low <= scene < high for scene in scenes):
            raise ValueError('Explicit scene indices exceed the declared dataset bounds')
    _DATASETS[spec.key]=spec
    if provider is not None:_PROVIDERS[spec.key]=provider


def get_dataset(key):
    try:return _DATASETS[key]
    except KeyError:raise ValueError(f'Unknown dataset {key}; installed integrations: {list(_DATASETS)}') from None


def datasets():
    return dict(_DATASETS)


def get_provider(key):
    get_dataset(key)
    provider=_PROVIDERS.get(key)
    if provider is None:raise ValueError(f'{key} has metadata but no executable dataset provider')
    if isinstance(provider,str):
        provider=importlib.import_module(provider)
        _PROVIDERS[key]=provider
    for method in ('preflight','frame_cloud','evaluate'):
        if not callable(getattr(provider,method,None)):raise ValueError(f'{key} provider is missing {method}')
    return provider


register_dataset(DatasetSpec('graspnet1b','GraspNet-1B',('realsense','kinect'),
    {'train':(0,100),'test_seen':(100,130),'test_similar':(130,160),'test_novel':(160,190)},
    256,('rgb','depth','intrinsics','poses','segmentation'), 'https://graspnet.net/datasets.html'),
    'grasppanda.integrations.graspnet1b')


register_dataset(DatasetSpec('dexgraspnet2','DexGraspNet 2.0',('realsense',),
    {'train':(0,8500),'test_seen':(100,130),'test_similar':(130,160),'test_novel':(160,190)},256,
    ('rendered_depth','intrinsics','poses','segmentation','leap_hand_targets'),
    'https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0',
    scene_splits={'train':tuple(range(100))+tuple(range(1000,8500))},
    method_actions={method:('infer','train_short','train') for method in
                    ('dexgraspnet2','dexgraspnet2_isa','dexgraspnet2_cvae')},
    protocol_note='Single rendered depth view and native instance-based workspace. Outputs retain camera-frame hand poses and 16 LEAP joint angles. The author GraspNet-derived scene protocol is registered; ACRONYM and larger-clutter evaluation suites remain separate native workflows. Prediction is not Isaac Gym success evaluation.',
    default_method='dexgraspnet2'), 'grasppanda.integrations.dexgraspnet2')


_gc6d = json.loads((Path(__file__).parent/'resources/graspclutter6d.json').read_text())
register_dataset(DatasetSpec('graspclutter6d', 'GraspClutter6D',
    ('realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'),
    {'train': (0, 1000), 'test': (0, 1000)}, 13,
    ('rgb', 'depth', 'intrinsics', 'poses', 'segmentation'),
    'https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D',
    scene_splits={name: tuple(ids) for name, ids in _gc6d['scene_splits'].items()},
    method_actions={'contact_graspnet_gc6d': ('infer', 'evaluate', 'train_short', 'train'),
                    'graspnet_baseline': ('infer', 'evaluate'), 'graspness': ('infer', 'evaluate')},
    protocol_note='official_gt_workspace uses the author 2D instance-mask bounding box with a 10% image margin; depth_only uses all valid depth pixels. GraspNet-trained checkpoints are cross-dataset transfer, not GC6D-trained benchmark results.',
    default_method='contact_graspnet_gc6d'),
    'grasppanda.integrations.graspclutter6d')


register_dataset(DatasetSpec('zerograsp11b', 'ZeroGrasp-11B', ('synthetic-rgbd',),
    {'train': (0, 10000)}, 100,
    ('rgb', 'depth', 'intrinsics', 'instance_masks', 'surface_points', 'grasp_labels'),
    'https://github.com/sh8/ZeroGrasp#download-zerograsp-dataset',
    method_actions={'zerograsp': ('infer', 'train_short', 'train')},
    protocol_note='The public training release contains 10,000 shards with 100 observations per shard. '
                  'Scene and frame configuration fields select shard and sample indices, not physical scenes. '
                  'Inference uses supplied visible instance masks and synthetic depth; no target shapes or grasps. '
                  'Training-set predictions are not held-out grasp AP.',
    default_method='zerograsp', scene_label='First shard', frame_label='First sample',
    hidden_controls=('num_points',)),
    'grasppanda.integrations.zerograsp11b')


register_dataset(DatasetSpec('jacquard', 'Jacquard', ('synthetic-rgbd',),
    {'train': (0, 54485), 'val': (0, 54485)}, 1,
    ('rgb', 'perfect_depth', 'planar_grasp_rectangles'),
    'https://jacquard.liris.cnrs.fr/database.php',
    method_actions={method: ('infer', 'train_short', 'train', 'evaluate')
                    for method in ('grconvnet_rgbd', 'grconvnet_depth')},
    protocol_note='Planar image-space grasps, not camera-frame 6-DoF poses. '
                  'The toolbox defaults to a deterministic object-disjoint train/validation split. '
                  'Published weights may have seen these observations; starter results are not held-out paper reproduction.',
    default_method='grconvnet_rgbd', scene_label='First sample in split', frame_label='Frame (always 0)',
    runner='grasppanda.methods.grconvnet', hidden_controls=('num_points','frame','collision_thresh')),
    'grasppanda.integrations.jacquard')


register_dataset(DatasetSpec('suctionnet1b', 'SuctionNet-1B', ('realsense',),
    {'train': (0, 100), 'test_seen': (100, 130), 'test_similar': (130, 160), 'test_novel': (160, 190)},
    256, ('rgb', 'depth', 'intrinsics', 'seal_labels', 'suction_collision_labels'),
    'https://graspnet.net/suction', scene_splits={'train':tuple(i for i in range(100) if i != 51)},
    method_actions={'suctionnet_rgbd':('infer', 'train_short', 'train')},
    protocol_note='Native suction scores, normals and positions from RGB-D. Shares GraspNet images, '
                  'but uses separate suction labels and output semantics. Training excludes scene 51 as in the author loader. '
                  'The registered weight and camera are RealSense; suction benchmark AP remains an upstream workflow.',
    default_method='suctionnet_rgbd', runner='grasppanda.methods.suctionnet',
    hidden_controls=('num_points','collision_thresh')),
    'grasppanda.integrations.suctionnet1b')


register_dataset(DatasetSpec('graspgen', 'GraspGen', ('synthetic-depth',),
    {'train':(0,8515),'valid':(0,8515)}, 1,
    ('object_mesh','rendered_depth','grasp_poses','simulation_labels'),
    'https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen',
    method_actions={'graspgen':('infer','train_short','train')},
    protocol_note='Object-centric partial-depth protocol with Franka Panda poses. Author train/valid UUID membership is preserved. '
                  'Generator and discriminator have separate training stages. The starter contains four objects; '
                  'training losses and prediction confidences are not simulated grasp success.',
    default_method='graspgen', scene_label='First object in split', frame_label='Frame (always 0)',
    runner='grasppanda.methods.graspgen', hidden_controls=('num_points','frame','collision_thresh')),
    'grasppanda.integrations.graspgen')


register_dataset(DatasetSpec('targo', 'TARGO', ('synthetic-depth',),
    {'train':(0,1000000),'val':(0,1000000),'test':(0,1000000)}, 1,
    ('depth','target_mask','scene_mask','intrinsics','extrinsics','visual_points','grasp_labels'),
    'https://huggingface.co/datasets/randing2000/TARGO',
    method_actions={'targonet':('infer','train_short','train')},
    protocol_note='Target-conditioned depth with supplied segmentation. Author shape completion and grasp networks retain '
                  'VGN workspace-frame pose conventions. Training/validation split groups every target and scene variant '
                  'by base scene ID; simulator success remains an upstream workflow.',
    default_method='targonet',scene_label='First scene in split',frame_label='Frame (always 0)',
    runner='grasppanda.methods.targo',hidden_controls=('num_points','frame','collision_thresh')),
    'grasppanda.integrations.targo')


register_dataset(DatasetSpec('acronym','ACRONYM',('synthetic-depth',),
    {'train':(0,1000000),'test':(0,1000000)},20,
    ('rendered_depth','intrinsics','object_mesh','grasp_labels'), 'https://github.com/NVlabs/acronym',
    method_actions={'graspldm':('infer','train_short','train')},
    protocol_note='Object-centric single-view depth. Native partial-cloud GraspLDM networks and losses; '
                  'toolbox seeded rendering and all-success-label supervision. Author visibility-filtered '
                  'training and simulator benchmark results are not reproduced.',
    default_method='graspldm',scene_label='First object in split',frame_label='First rendered view (0–19)',
    runner='grasppanda.methods.graspldm',hidden_controls=('num_points','collision_thresh')),
    'grasppanda.integrations.acronym')
