"""Target-conditioned depth inputs and scene-disjoint TARGO training contracts."""
import csv
import math
from pathlib import Path
import random
import re

DEFAULTS = {'train_fraction':.9,'split_seed':0,'quality_threshold':.9,'outside_threshold':.2,'force_detection':True}
FIELDS = ('scene_id','qx','qy','qz','qw','x','y','z','width','label')
INVALID_SCENES = {'b960209b0cbd406d98dac25aeccd3c71_s_2','5522fdbe62d9450687a195cfd2bbbac3_c_1','b960209b0cbd406d98dac25aeccd3c71_c_2'}
_LABEL_IDS = {}
_SCENES = {}


def options(config):
    return {**DEFAULTS,**config.dataset_options}


def validate_options(config):
    values = options(config)
    if set(values) != set(DEFAULTS): raise ValueError('TARGO dataset_options accepts '+', '.join(DEFAULTS))
    for key,low,high in (('train_fraction',.05,.95),('quality_threshold',0,1),('outside_threshold',.01,.99)):
        value = values[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid TARGO '+key)
    if type(values['split_seed']) is not int or not 0 <= values['split_seed'] < 2**32:
        raise ValueError('TARGO split_seed must be an integer in [0, 2**32)')
    if type(values['force_detection']) is not bool: raise ValueError('TARGO force_detection must be boolean')
    if config.trainer: raise ValueError('TARGO uses its native grasp objective; no trainer overrides are registered')
    if config.workspace != 'target_depth' or config.num_points != 2048 or config.collision_thresh != 0:
        raise ValueError('TARGO requires target_depth, num_points: 2048 and collision_thresh: 0; native TSDF filtering is retained')
    if config.frame or config.checkpoint_policy != 'strict' or config.sdf_root:
        raise ValueError('TARGO requires frame: 0, strict checkpoints and an empty sdf_root')
    if config.label_root or config.data_workers:
        raise ValueError('TARGO labels are in syn_train/grasps.csv; leave label_root empty and data_workers: 0')
    if config.action in ('train','train_short') and config.split != 'train':
        raise ValueError('TARGO training uses split: train')


def scene_index(directory):
    directories = [directory]+sorted(p for p in directory.glob('*') if p.is_dir())
    identity = tuple((str(p),p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in directories if p.exists())
    if directory in _SCENES and _SCENES[directory][0] == identity: return _SCENES[directory][1]
    rows = {}
    for path in sorted(p for folder in directories for p in folder.glob('*.npz')):
        if not re.fullmatch(r'[0-9a-f]{32}_[csd]_\d+',path.stem): continue
        if path.stem in rows: raise ValueError('Duplicate TARGO scene: '+path.stem)
        if not path.resolve().is_relative_to(directory.resolve()): raise ValueError('TARGO scene resolves outside its dataset root')
        rows[path.stem] = path
    _SCENES[directory] = (identity,rows)
    return rows


def label_ids(labels):
    stat = labels.stat()
    identity = (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if labels in _LABEL_IDS and _LABEL_IDS[labels][0] == identity: return _LABEL_IDS[labels][1]
    ids = set()
    with labels.open() as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS: raise ValueError('Unexpected TARGO grasp CSV columns')
        for row in reader:
            uid = row['scene_id']
            if not re.fullmatch(r'[0-9a-f]{32}_[csd]_\d+',uid): raise ValueError('Invalid TARGO scene ID')
            if uid not in INVALID_SCENES: ids.add(uid)
    _LABEL_IDS[labels] = (identity,ids)
    return ids


def inventory(config,split=None):
    root = Path(config.dataset_root)
    split = split or config.split
    if not config.dataset_root or not root.is_dir(): raise ValueError('Choose a TARGO root; use Download starter data')
    if split == 'test':
        result = scene_index(root/'test_set_gaussian_0.005/scenes')
        if not result: raise ValueError('No TARGO test scenes; expected test_set_gaussian_0.005/scenes/*.npz')
        return [{'uid':uid,'path':path} for uid,path in result.items()]
    labels = root/'syn_train/grasps.csv'
    if not labels.is_file(): raise ValueError('Missing TARGO syn_train/grasps.csv; download the training release or starter')
    ids = label_ids(labels)
    groups = sorted({uid.split('_')[0] for uid in ids})
    if len(groups) < 2: raise ValueError('TARGO scene-disjoint validation requires at least two base scenes')
    settings = options(config)
    random.Random(settings['split_seed']).shuffle(groups)
    boundary = max(1,min(len(groups)-1,int(len(groups)*settings['train_fraction'])))
    train = set(groups[:boundary])
    chosen = sorted(uid for uid in ids if (uid.split('_')[0] in train) == (split=='train'))
    files = scene_index(root/'syn_train/scenes')
    missing = set(chosen)-files.keys()
    if missing: raise ValueError('TARGO split references uninstalled scene '+sorted(missing)[0])
    test_groups = {uid.split('_')[0] for uid in scene_index(root/'test_set_gaussian_0.005/scenes')}
    if test_groups & set(groups): raise ValueError('TARGO training and test base scenes overlap')
    return [{'uid':uid,'path':files[uid]} for uid in chosen]


def selection(config,rows):
    if config.action == 'train' and not config.train_batch_limit: return rows
    if config.scene+config.frames > len(rows):
        raise ValueError(f'TARGO {config.split} has {len(rows)} scenes; reduce the first scene or scene count')
    return rows[config.scene:config.scene+config.frames]


def assets(config):
    from ..config import ROOT
    from ..weights import records
    return {r['path']:ROOT/r['path'] for r in records('targonet','synthetic-depth') if r.get('role')=='auxiliary'}


def preflight(config):
    validate_options(config)
    selection(config,inventory(config))
    if config.action == 'train': inventory(config,'val')
    if config.action == 'infer' and not config.checkpoint: raise ValueError('TARGO inference requires a trained grasp checkpoint')
    if config.checkpoint and not Path(config.checkpoint).is_file(): raise ValueError('Missing TARGO checkpoint; use Download registered weights')
    for path in assets(config).values():
        if not path.is_file(): raise ValueError('Missing TARGO shape-completion weights; use Download registered weights')
    setup = Path(config.dataset_root)/('test_set_gaussian_0.005' if config.split=='test' else 'syn_train')/'setup.json'
    import json
    if not setup.is_file() or abs(json.loads(setup.read_text()).get('size',0)-.3)>1e-6:
        raise ValueError('TARGO requires the author setup.json with a 0.3 m workspace')


def protocol(config):
    return dict(dataset=config.dataset,method=config.method,split=config.split,dataset_options=options(config),
                input='single-view depth-derived scene points and a supplied target segmentation mask',
                coordinate_frame='author 0.3 m workspace',translation_units='metres',
                pose_convention='VGN gripper pose; xyzw quaternions',
                inference='frozen author AdaPoinTr completion, native TARGO-Net and TSDF filtering',
                training='author released target TSDF points and scene points; native quality, symmetric rotation and width losses',
                validation='base-scene-disjoint labelled loss; not simulator grasp success')


def frame_cloud(config,scene,frame):
    raise ValueError('TARGO uses its target-conditioned native point-cloud runner')


def evaluate(config,out):
    raise ValueError('TARGO simulation evaluation remains an upstream workflow; epoch validation reports labelled losses only')
