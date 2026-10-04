"""Explicit ACRONYM object splits and partial-depth GraspLDM contracts."""
import json
import math
from pathlib import Path, PurePosixPath
import re

VIEWS = 20
DEFAULTS = dict(mode='diffusion',sampler='ddim',inference_steps=100,num_grasps=20)
PATTERN = r'[A-Za-z0-9]+_[0-9a-f]{1,32}_[0-9.eE+-]+\.h5'


def options(config): return {**DEFAULTS,**config.dataset_options}


def trainer(config): return {'stage':'vae','grasps_per_view':100,'supervision':'all_success',**config.trainer}


def validate_options(config):
    if not isinstance(config.dataset_options,dict) or not isinstance(config.trainer,dict):
        raise ValueError('ACRONYM dataset_options and trainer must be mappings')
    values = options(config)
    if set(values)!=set(DEFAULTS): raise ValueError('ACRONYM dataset_options accepts '+', '.join(DEFAULTS))
    if values['mode'] not in ('vae','diffusion') or values['sampler'] not in ('ddim','ddpm'):
        raise ValueError('Choose mode: vae or diffusion, and sampler: ddim or ddpm')
    for name,low,high in (('inference_steps',1,1000),('num_grasps',1,1000)):
        if type(values[name]) is not int or not low<=values[name]<=high:raise ValueError('Invalid ACRONYM '+name)
    if values['sampler']=='ddpm' and values['inference_steps']!=1000:
        raise ValueError('The native DDPM sampling path requires inference_steps: 1000; use DDIM for fewer steps')
    setting = trainer(config)
    if set(setting)!={'stage','grasps_per_view','supervision'} or setting['stage'] not in ('vae','diffusion'):
        raise ValueError('GraspLDM trainer accepts stage (vae or diffusion), grasps_per_view and supervision')
    if type(setting['grasps_per_view']) is not int or not 2<=setting['grasps_per_view']<=1000:
        raise ValueError('grasps_per_view must be in [2, 1000]')
    if setting['supervision']!='all_success':
        raise ValueError('Only all_success supervision is registered; author visibility-filtered training is not reproduced')
    if config.trainer and config.action not in ('train','train_short'):raise ValueError('GraspLDM trainer parameters apply only to training')
    if config.workspace!='acronym_partial' or config.num_points!=1024 or config.collision_thresh!=0:
        raise ValueError('ACRONYM requires acronym_partial, num_points: 1024 and collision_thresh: 0')
    if config.checkpoint_policy!='strict' or config.sdf_root:
        raise ValueError('GraspLDM requires strict checkpoints and an empty sdf_root')
    if config.data_workers or config.eval_batch_limit:
        raise ValueError('ACRONYM renders in its GPU worker; data_workers and eval_batch_limit must be 0. No benchmark evaluator is registered.')
    if config.action in ('train','train_short') and config.split!='train':raise ValueError('ACRONYM training requires split: train')


def split_files(root):
    custom = root/'splits.json'
    if custom.is_file(): splits=json.loads(custom.read_text())
    else:
        splits={'train':[],'test':[]}
        files=sorted((root/'splits').glob('*.json'))
        if not files:raise ValueError('Missing ACRONYM splits.json or author splits/*.json; download starter data or follow Guide → Datasets')
        for path in files:
            value=json.loads(path.read_text())
            for key in splits:splits[key].extend(name.removesuffix('.json')+'.h5' if name.endswith('.json') else name for name in value[key])
    if set(splits)!={'train','test'}:raise ValueError('ACRONYM splits require exactly train and test lists')
    for key,values in splits.items():
        if not isinstance(values,list) or not values or any(not isinstance(name,str) or not re.fullmatch(PATTERN,name) for name in values):
            raise ValueError('ACRONYM split lists must contain original HDF5 basenames')
        splits[key]=sorted(set(values))
    identities=lambda names:{name.split('_')[1].zfill(32) for name in names}
    if identities(splits['train']) & identities(splits['test']):raise ValueError('ACRONYM train/test splits share a base mesh identity')
    return splits


def inventory(config,split=None):
    root=Path(config.dataset_root)
    if not config.dataset_root or not root.is_dir():raise ValueError('Choose an ACRONYM root; use Download starter data')
    rows=[]
    for name in split_files(root)[split or config.split]:
        path=root/'grasps'/name
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Missing or non-local ACRONYM annotation: '+name)
        rows.append(dict(uid=name[:-3],path=path))
    return rows


def selection(config,rows):
    if config.action=='train' and not config.train_batch_limit:
        return [(row,view) for row in rows for view in range(VIEWS)]
    start=config.scene*VIEWS+config.frame
    if start+config.frames>len(rows)*VIEWS:raise ValueError(f'ACRONYM {config.split} has {len(rows)} objects with {VIEWS} selectable views each')
    return [(rows[i//VIEWS],i%VIEWS) for i in range(start,start+config.frames)]


def object_info(config,row):
    import h5py
    with h5py.File(row['path'],'r') as data:
        if any(isinstance(data.get(key,getlink=True),h5py.ExternalLink) for key in ('object','object/file','object/scale')):
            raise ValueError('External HDF5 object links are not supported')
        name=data['object/file'][()]
        name=name.decode() if isinstance(name,bytes) else str(name)
        scale=float(data['object/scale'][()])
    path=PurePosixPath(name)
    root=Path(config.dataset_root).resolve();mesh=root/path
    if path.is_absolute() or '..' in path.parts or not mesh.resolve().is_relative_to(root) or not mesh.is_file():
        raise ValueError('Missing or non-local ACRONYM mesh: '+name)
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid ACRONYM object scale')
    return mesh,scale


def assets(config):
    from ..config import ROOT
    from ..weights import records
    return {r['path']:ROOT/r['path'] for r in records('graspldm','synthetic-depth') if r.get('role')=='auxiliary'}


def preflight(config):
    validate_options(config)
    rows=selection(config,inventory(config))
    for row in {row['uid']:row for row,_ in rows}.values():object_info(config,row)
    if config.action=='infer' and not config.checkpoint:raise ValueError('GraspLDM prediction requires a trained checkpoint')
    if config.checkpoint and not Path(config.checkpoint).is_file():raise ValueError('Missing GraspLDM checkpoint; use Download registered weights')
    for path in assets(config).values():
        if not path.is_file():raise ValueError('Missing GraspLDM VAE initializer; use Download registered weights')


def protocol(config):
    return dict(dataset=config.dataset,method=config.method,split=config.split,input='1024 points from one rendered isolated-object depth view',
                coordinate_frame='OpenCV camera; x right, y down, z forward',translation_units='metres',
                pose_convention='native ACRONYM Franka gripper base; no predicted opening width',
                rendering='toolbox seeded view on a 0.5–0.8 m shell, 640×480 author camera calibration; no grasp-label view selection',
                normalization='author partial-cloud mean centering, XYZ / 0.05 and modified Rodrigues parameters / 0.5',
                training='native VAE/diffusion objectives with original successful grasps; toolbox all_success supervision, without unavailable author visibility filtering',
                evaluation='no simulation success or paper benchmark reproduction',dataset_options=options(config))


def frame_cloud(config,scene,frame):raise ValueError('ACRONYM uses its isolated-object native runner')


def evaluate(config,out):raise ValueError('ACRONYM simulation evaluation remains an upstream workflow')
