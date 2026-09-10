"""Configurable pose refinement with independent, frozen auxiliary networks."""
from pathlib import Path
import json
import subprocess
import sys

METHODS = ('generalizing_grasp', 'graspnet_baseline', 'graspness', 'scale_balanced_grasp', 'fgc_graspnet',
           'pointnet2_upgrade', 'graspfast', 'graspbalance', 'granet', 'graspness_modern',
           'economicgrasp', 'dograspnet', 'hggd', 'region_normalized_grasp', 'finegrasp', 'gtg2')
SCHEMA = {
    'iterations': ('int', 1, 2000),
    'top_per_instance': ('int', 1, 100),
    'scene_points': ('int', 2048, 50000),
    'min_object_points': ('int', 64, 2048),
    'gripper_points': ('int', 16, 2048),
    'nms_translation': ('float', .001, .1),
    'nms_angle': ('float', 1., 180.),
    'max_assignment_distance': ('float', .001, .2),
    'min_width': ('float', .0001, .1),
    'max_width': ('float', .001, .2),
    'acceptance': ('choice', ('joint', 'total')),
    'coordinate_frame': ('choice', ('table', 'camera')),
    'output': ('choice', ('all', 'selected')),
    'unassigned': ('choice', ('keep', 'drop', 'error')),
    'contact_checkpoint': ('checkpoint_path',),
    'score_checkpoint': ('checkpoint_path',),
    'segmentation_checkpoint': ('checkpoint_path',),
    **{name+'_lr': ('float', 0., .01) for name in ('translation', 'approach', 'angle', 'width', 'depth')},
    **{name+'_weight': ('float', 0., 100.) for name in ('contact_map', 'projection_map', 'center', 'contact_distance', 'score')},
}
WEIGHTS = {'contact_checkpoint': 'generalizing_contactnet', 'score_checkpoint': 'generalizing_scorenet',
           'segmentation_checkpoint': 'scale_balanced_dsn'}


def enabled(config):
    return bool(config.refinement)


def validate(config):
    from ..module_options import validate_options
    if not isinstance(config.refinement, dict):raise ValueError('refinement must be a mapping')
    if not enabled(config):return
    if config.refinement.get('type')!='contact_score':raise ValueError('refinement requires type: contact_score')
    if config.method not in METHODS or config.action not in ('infer', 'evaluate'):
        raise ValueError('Contact-score refinement requires a registered 6-DoF inference/evaluation adapter')
    options = {k:v for k,v in config.refinement.items() if k not in ('type', 'segmentation')}
    validate_options(config.method, 'refinement', 'contact_score', options)
    segmentation = config.refinement.get('segmentation', {})
    if not isinstance(segmentation,dict):raise ValueError('refinement.segmentation must be a mapping')
    from ..methods.scale_balanced_sampling import SCHEMA as dsn
    forbidden = set(segmentation)-set(dsn) | (set(segmentation)&{'seed_count','empty_policy','segmentation_checkpoint'})
    if forbidden:raise ValueError('Unsupported segmentation parameters: '+str(sorted(forbidden)))
    validate_options('scale_balanced_grasp', 'sampling', 'object_balanced', segmentation)
    if config.camera!='realsense' and not options.get('segmentation_checkpoint'):
        raise ValueError('Contact-score refinement requires a matching DSN checkpoint for Kinect')
    if options.get('min_width',.001)>=options.get('max_width',.1):raise ValueError('min_width must be smaller than max_width')
    rates = dict(translation=.0002,approach=.002,angle=.002,width=.0001,depth=0.)
    if not any(options.get(k+'_lr',v)>0 for k,v in rates.items()):raise ValueError('At least one refinement learning rate must be positive')
    if not any(options.get(k+'_weight',v)>0 for k,v in dict(contact_map=1.,projection_map=.2,center=5.,contact_distance=1.,score=.1).items()):
        raise ValueError('At least one refinement objective must have a positive weight')


def artifacts(config):
    """Resolve content-checked weights and the exact auxiliary source files."""
    from ..config import ROOT, catalogue
    from ..weights import component_records
    from ..jobs import digest
    if not enabled(config):return dict(weights={}, sources={})
    registry=component_records();weights={};sources={}
    for key, ident in WEIGHTS.items():
        row=registry[ident];custom=config.refinement.get(key)
        path=(ROOT/Path(custom or row['path']).expanduser()).resolve()
        if not path.is_file():raise ValueError('Refinement weight missing: run ./panda component-weights '+ident+' or set refinement.'+key)
        sha=digest(path)
        if not custom and sha!=row['sha256']:raise ValueError('Registered refinement weight checksum mismatch: '+ident)
        weights[str(path)]=sha
    pins={r['id']:r.get('pinned_commit',r.get('commit')) for r in json.loads((ROOT/'grasppanda/resources/upstreams.lock.json').read_text())}
    for name in ('generalizing_grasp', 'scale_balanced_grasp'):
        repo=ROOT/catalogue()[name]['path']
        if not (repo/'.git').exists():raise ValueError('Refinement source missing; run ./panda fetch')
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if commit!=pins[name]:raise ValueError('Refinement source revision differs from its pin: '+name)
        sources.update({str(p.resolve()):digest(p) for p in repo.rglob('*.py')})
    return dict(weights=weights,sources=sources)


def weight_paths(config):
    from ..config import ROOT
    from ..weights import component_records
    return {k:str((ROOT/Path(config.refinement.get(k) or component_records()[v]['path']).expanduser()).resolve()) for k,v in WEIGHTS.items()}


def run(config, out, result):
    """Run auxiliary namespaces in their own process within the shared environment."""
    from ..config import ROOT
    import time
    if not enabled(config):return result
    start=time.monotonic()
    input_path=out/'refinement-input.json';input_path.write_text(json.dumps(result,indent=2)+'\n')
    subprocess.run([sys.executable,'-m','grasppanda.refinement.pipeline',str(out/'config.json'),str(out)],cwd=ROOT,check=True)
    result=json.loads((out/'refinement-result.json').read_text())
    result['refinement']['wall_seconds']=time.monotonic()-start
    return result


def preflight(config):
    if not enabled(config):return
    root=Path(config.dataset_root)/'scenes'
    for index in range(config.scene*256+config.frame,config.scene*256+config.frame+config.frames):
        scene,frame=divmod(index,256)
        folder=root/f'scene_{scene:04d}'/config.camera
        needed=[]
        if config.method=='generalizing_grasp':needed=[folder/'cam0_wrt_table.npy']
        elif config.refinement.get('coordinate_frame','table')=='table':needed=[folder/'cam0_wrt_table.npy',folder/'camera_poses.npy']
        if any(not p.is_file() for p in needed):raise ValueError('Refinement needs camera-to-table calibration: '+str(next(p for p in needed if not p.is_file())))


def settings(value):
    """Compare processing settings independently of machine-specific weight paths."""
    return {k:v for k,v in value.items() if k not in WEIGHTS}


def manifest_artifacts(config):
    """Portable source and weight identities for exported predictions."""
    from ..config import ROOT,catalogue
    record=artifacts(config)
    paths=weight_paths(config)
    weights={name:record['weights'][path] for name,path in paths.items()}
    sources={}
    for name in ('generalizing_grasp','scale_balanced_grasp'):
        root=(ROOT/catalogue()[name]['path']).resolve()
        for path,sha in record['sources'].items():
            file=Path(path)
            if file.is_relative_to(root):sources[name+'/'+str(file.relative_to(root))]=sha
    return dict(weights=weights,sources=sources)
