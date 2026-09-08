"""Dataset registry with explicit executable providers and metadata."""
from dataclasses import dataclass
import importlib


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    cameras: tuple[str,...]
    splits: dict[str,tuple[int,int]]
    frames_per_scene: int
    modalities: tuple[str,...]
    download_url: str

_DATASETS={}
_PROVIDERS={}


def register_dataset(spec,provider=None):
    if spec.key in _DATASETS:raise ValueError(f'Dataset already registered: {spec.key}')
    if not spec.key or spec.frames_per_scene<=0:raise ValueError('Invalid dataset specification')
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
