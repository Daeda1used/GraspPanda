"""Native VGN simulation-data and voxel-observation contracts."""
import json
import math
from pathlib import Path
import re

ASSET_SHA = '00645654bebeddd82d6309c52e4808a09577c0e56bf64cc72a76a9fc24578029'
DEFAULTS = dict(scene_type='pile',scene_count=3,grasps_per_scene=120,max_views=6,
                train_fraction=.9,split_seed=0,augment=False,quality_threshold=.9,
                simulation_rounds=5,simulation_objects=5,simulation_views=6)


def options(config):return {**DEFAULTS,**config.dataset_options}


def generation_plan(config):
    from importlib.metadata import version
    from ..runtime.vgn_source import COMMIT
    settings=options(config)
    return dict(source=COMMIT,adapter=1,assets=ASSET_SHA,seed=config.seed,
                runtime={name:version(name) for name in ('pybullet','open3d','numpy')},
                settings={k:settings[k] for k in ('scene_type','scene_count','grasps_per_scene','max_views','train_fraction','split_seed')})


def validate_options(config):
    if not isinstance(config.dataset_options,dict) or set(config.dataset_options)-set(DEFAULTS):
        raise ValueError('VGN dataset_options accepts '+', '.join(DEFAULTS))
    values=options(config)
    if values['scene_type'] not in ('pile','packed'):raise ValueError('VGN scene_type must be pile or packed')
    for key,low,high in (('scene_count',2,1000000),('grasps_per_scene',1,10000),('max_views',1,6),
                         ('split_seed',0,2**31-1),('simulation_rounds',1,10000),('simulation_objects',1,10),('simulation_views',1,12)):
        if type(values[key]) is not int or not low<=values[key]<=high:raise ValueError('Invalid VGN '+key)
    for key in ('train_fraction','quality_threshold'):
        if type(values[key]) not in (int,float) or not math.isfinite(values[key]) or not 0<values[key]<1:raise ValueError('VGN '+key+' must be strictly between 0 and 1')
    if type(values['augment']) is not bool:raise ValueError('VGN augment must be a boolean')
    if config.workspace!='vgn_tsdf' or config.collision_thresh!=0 or config.num_points!=2048:
        raise ValueError('VGN requires vgn_tsdf, collision_thresh: 0 and the unused num_points field fixed at 2048')
    if config.checkpoint_policy!='strict' or config.label_root or config.sdf_root or config.data_workers or config.trainer:
        raise ValueError('VGN uses strict loading, its dataset-local cache, data_workers: 0 and no trainer overrides')
    if config.action in ('train','train_short') and config.split!='train':raise ValueError('VGN training requires split: train')
    if config.action=='generate' and config.checkpoint:raise ValueError('Simulation data generation does not use weights; set checkpoint to an empty string')


def inventory(config,split=None):
    root=Path(config.dataset_root);path=root/'splits.json'
    if not path.is_file():raise ValueError('VGN data is not generated; select Generate simulation data first')
    splits=json.loads(path.read_text())
    if set(splits)!= {'train','val'} or any(not isinstance(v,list) or not v for v in splits.values()):raise ValueError('VGN splits require nonempty train and val lists')
    flat=[name for values in splits.values() for name in values]
    if any(not isinstance(s,str) or not re.fullmatch(r'scene_[0-9]{8}',s) for s in flat) or len(flat)!=len(set(flat)):
        raise ValueError('Invalid or overlapping VGN scene splits')
    for name in splits[split or config.split]:
        for directory in ('raw/scenes','processed/scenes'):
            path=root/directory/(name+'.npz')
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Missing or external VGN observation: '+str(path))
    return splits[split or config.split]


def selection(config,rows):
    if config.action=='train' and not config.train_batch_limit:return rows
    if config.scene+config.frames>len(rows):raise ValueError(f'VGN {config.split} contains {len(rows)} scenes; reduce scene or frames')
    return rows[config.scene:config.scene+config.frames]


def preflight(config):
    from ..jobs import digest
    validate_options(config)
    root=Path(config.dataset_root)
    if not config.dataset_root or not root.is_dir():raise ValueError('Choose a VGN dataset root and download the author assets first')
    if config.action in ('generate','simulate'):
        archive=root/'assets/vgn-data.zip'
        if not archive.is_file() or digest(archive)!=ASSET_SHA:raise ValueError('Missing or changed VGN assets; use Download starter data')
        if config.action=='generate' and (root/'generation.json').is_file():
            if json.loads((root/'generation.json').read_text())!=generation_plan(config):
                raise ValueError('This root has another generation plan or runtime; use its original settings or choose a new dataset root')
    else:
        selection(config,inventory(config))
        if not (root/'processed/grasps.csv').is_file() or not (root/'raw/setup.json').is_file():raise ValueError('Incomplete VGN dataset; rerun its generation job')
    if config.action in ('infer','simulate') and not config.checkpoint:raise ValueError('VGN prediction requires trained weights; use Download registered weights')
    if config.checkpoint and not Path(config.checkpoint).is_file():raise ValueError('Missing VGN checkpoint; download weights or clear checkpoint for training from scratch')


def protocol(config):
    path=Path(config.dataset_root)/'generation.json'
    return dict(dataset=config.dataset,method=config.method,input='40 x 40 x 40 TSDF integrated from rendered depth only',
                frame='native VGN workspace',units='metres; xyzw quaternion',pose='native VGN gripper TCP convention',
                data='locally generated native PyBullet grasp trials, not a downloaded large benchmark',
                splitting='toolbox scene-disjoint train/val split; simulation uses author held-out test objects',
                objectives='native quality BCE, symmetric quaternion loss and width MSE',
                dataset_options=options(config),
                generation=json.loads(path.read_text()) if path.is_file() and config.action!='simulate' else None)


def frame_cloud(config,scene,frame):raise ValueError('VGN uses its native voxel runner')


def evaluate(config,out):raise ValueError('Select Simulate clutter removal for VGN physics evaluation')
