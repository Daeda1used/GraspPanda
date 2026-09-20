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

    @property
    def inference_split(self):
        return next((name for name in ('test_seen', 'test', 'val') if name in self.splits), next(iter(self.splits)))

    def scene_ids(self, split):
        if split not in self.splits:
            raise ValueError(f'Unknown {self.title} split: {split}')
        if split in self.scene_splits:
            return self.scene_splits[split]
        return tuple(range(*self.splits[split]))

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
    default_method='zerograsp', scene_label='First shard', frame_label='First sample'),
    'grasppanda.integrations.zerograsp11b')
