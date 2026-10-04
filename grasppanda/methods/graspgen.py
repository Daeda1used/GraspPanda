"""Native GraspGen networks with bounded data, paired weights and stage training."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

from ..config import ROOT, catalogue
from ..integrations.graspgen import inventory, selection, layout, protocol, trainer, GraspLabelReader
from ..jobs import digest
from ..training.state import capture_rng_state, restore_rng_state, same_state, write_result


def prepare():
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    repo = ROOT/catalogue()['graspgen']['path']
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    # Author gripper assets are relative to the source root. Workers are isolated processes.
    os.chdir(repo)
    import numpy as np
    import trimesh
    from grasp_gen.dataset import dataset as source, renderer
    if getattr(renderer,'_grasppanda_prepared',False): return

    class SeededRandom:
        def __getattr__(self,name): return getattr(np.random,name)
        def default_rng(self,seed=None):
            return np.random.default_rng(np.random.randint(0,2**32) if seed is None else seed)

    class SeededNumpy:
        random = SeededRandom()
        def __getattr__(self,name): return getattr(np,name)

    class MeshLoader:
        def __getattr__(self,name): return getattr(trimesh,name)
        def load(self,*args,**kwargs):
            value = trimesh.load(*args,**kwargs)
            return value.dump(concatenate=True) if isinstance(value,trimesh.Scene) else value

    renderer.np = SeededNumpy()
    render = renderer.render_point_cloud_from_object
    renderer.render_point_cloud_from_object = lambda data,count: render(data,count,prob_object_only=1.)
    source.trimesh = MeshLoader()
    renderer._grasppanda_prepared = True


@contextmanager
def observation_seed(seed):
    import numpy as np
    import torch
    state = capture_rng_state()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try: yield
    finally: restore_rng_state(state)


def sample_seed(config,uid,epoch=0,attempt=0):
    value = f'{config.seed}:{uid}:{epoch}:{attempt}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4],'little')


def build(config):
    import torch
    from grasp_gen.grasp_server import load_grasp_cfg, GraspGenSampler
    from grasp_gen.models.grasp_gen import GraspGen
    from ..weights import records
    rows = records('graspgen','synthetic-depth')
    architecture = next(r for r in rows if r['path'].endswith('.yml'))
    config_path = ROOT/architecture['path']
    if digest(config_path) != architecture['sha256']:
        raise ValueError('GraspGen architecture differs from the registered author configuration')
    cfg = load_grasp_cfg(str(config_path))
    cfg.meshcat.visualize = False
    cfg.discriminator.checkpoint_object_encoder_pretrained = None
    model = GraspGen.from_config(cfg.diffusion,cfg.discriminator)
    payload = {}
    if config.checkpoint:
        loaded = torch.load(config.checkpoint,map_location='cpu',weights_only=True)
        if loaded.get('format') == 'grasppanda-graspgen-v1':
            payload = loaded
            model.grasp_generator.load_state_dict(loaded['generator'],strict=True)
            model.grasp_discriminator.load_state_dict(loaded['discriminator'],strict=True)
        elif 'model' in loaded:
            model.grasp_generator.load_state_dict(loaded['model'],strict=True)
            secondary = torch.load(cfg.discriminator.checkpoint,map_location='cpu',weights_only=True)
            model.grasp_discriminator.load_state_dict(secondary['model'],strict=True)
        else:
            raise ValueError('Expected an author generator checkpoint or a paired GraspPanda GraspGen checkpoint')
    # Use the native sampler with safely loaded native modules.
    sampler = GraspGenSampler.__new__(GraspGenSampler)
    sampler.cfg, sampler.model = cfg, model.cuda().eval()
    return sampler, cfg, payload


def visual_observation(config,row,reader):
    """Rendering uses mesh geometry and scale alone, never grasp success labels."""
    import numpy as np
    import trimesh
    from grasp_gen.dataset.renderer import render_pc
    value = reader.read_grasps_by_uuid(row['uid'])
    scale = float(value['object']['scale'])
    if not np.isfinite(scale) or scale <= 0: raise ValueError('Invalid GraspGen object scale')
    mesh = trimesh.load(row['mesh'])
    if isinstance(mesh,trimesh.Scene): mesh = mesh.dump(concatenate=True)
    mesh.apply_scale(scale)
    asset = SimpleNamespace(object_mesh=mesh,object_asset_path=str(row['mesh']),object_scale=scale)
    for attempt in range(8):
        with observation_seed(sample_seed(config,row['uid'],attempt=attempt)):
            result, error = render_pc(asset,config.num_points,mesh_mode=False)
        if not result.get('invalid'):
            points = result['points'].numpy()
            if points.shape != (2048,3) or not np.isfinite(points).all():
                raise ValueError('Invalid rendered point cloud')
            return points,result['T_move_to_pc_mean'],attempt
    raise ValueError(f'Unable to render a usable partial view of {row["uid"]}: {error}')


def preview(points,poses,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from grasp_gen.robot import get_gripper_info
    from grasp_gen.utils.meshcat_utils import load_visualization_gripper_points
    fig = plt.figure(figsize=(8,6),layout='constrained')
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*points.T,s=2,c=points[:,2],cmap='viridis',alpha=.65)
    geometry = [points]
    controls = load_visualization_gripper_points('franka_panda')
    for pose in poses[:5]:
        for vertices in controls:
            segment = (pose@vertices)[:3].T
            geometry.append(segment)
            ax.plot(*segment.T,color='#d76548',linewidth=1.4)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Partial depth observation · native predicted grasps')
    bounds = np.concatenate(geometry)
    extent = np.ptp(bounds,axis=0).max()*.55
    center = (bounds.min(0)+bounds.max(0))/2
    ax.set_xlim(center[0]-extent,center[0]+extent)
    ax.set_ylim(center[1]-extent,center[1]+extent)
    ax.set_zlim(center[2]-extent,center[2]+extent)
    ax.set_box_aspect((1,1,1)); fig.savefig(out/'preview.png',dpi=130); plt.close(fig)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(points,colors=[110,135,151,255]),node_name='observed_depth')
    native_mesh = get_gripper_info('franka_panda').visual_mesh
    for index,pose in enumerate(poses[:3]):
        mesh = native_mesh.copy(); mesh.apply_transform(pose)
        scene.add_geometry(mesh,node_name=f'predicted_gripper_{index}')
    scene.export(out/'grasp_scene.glb')


def infer(config,out):
    import numpy as np
    import torch
    from grasp_gen.grasp_server import GraspGenSampler
    sampler, _, _ = build(config)
    target = out/'predictions'; target.mkdir(exist_ok=True)
    records = []
    rows = selection(config,inventory(config))
    reader = GraspLabelReader(config,rows)
    for row in rows:
        points, transform, retries = visual_observation(config,row,reader)
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.inference_mode():
            poses,scores = GraspGenSampler.run_inference(points,sampler,grasp_threshold=-1,
                num_grasps=200,topk_num_grasps=100,min_grasps=1,max_tries=1)
        torch.cuda.synchronize(); elapsed = time.monotonic()-start
        poses,scores = poses.cpu().numpy(),scores.cpu().numpy()
        if poses.ndim != 3 or poses.shape[1:] != (4,4) or scores.shape != (len(poses),):
            raise ValueError('Native GraspGen decoder returned unexpected shapes')
        if not np.isfinite(poses).all() or not np.isfinite(scores).all():
            raise ValueError('Non-finite GraspGen prediction')
        if len(poses) and (not np.allclose(poses[:,:3,:3]@poses[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-4)
                          or not np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-4)):
            raise ValueError('GraspGen returned invalid rotations')
        name = row['uid']+'.npz'
        np.savez_compressed(target/name,points=points,poses=poses,scores=scores,
                            object_to_observation=transform)
        records.append({'object':row['uid'],'prediction':name,'grasps':len(poses),
                        'sha256':digest(target/name),'mesh_sha256':digest(row['mesh']),
                        'render_retries':retries,'forward_decode_seconds':elapsed})
        if len(records) == 1: preview(points,poses,out)
    result = {'stage':'dataset_inference','dataset':config.dataset,'method':config.method,
              'frames':records,'prediction_dir':'predictions','protocol':protocol(config),
              'checkpoint_sha256':digest(config.checkpoint),'ap':None,
              'note':'Native confidence-ranked poses. Visual rendering uses no grasp-label filtering; simulation success is not measured.'}
    (target/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def native_dataset(config,cfg,out,rows):
    from grasp_gen.dataset.dataset import ObjectPickDataset, PickDataset
    root,labels,meshes = layout(config)
    cfg.data.root_dir = str(root)
    cfg.data.cache_dir = config.label_root or str(out/'native-cache')
    cfg.data.object_root_dir = str(meshes); cfg.data.grasp_root_dir = str(labels)
    cfg.data.dataset_name = 'objaverse'; cfg.data.dataset_version = 'v2'
    cfg.data.num_points = config.num_points; cfg.data.prob_point_cloud = 1.
    cfg.data.num_grasps_per_object = trainer(config)['grasps_per_object']
    cfg.data.load_discriminator_dataset = trainer(config)['stage'] == 'discriminator'
    cfg.data.discriminator_ratio = [.5,.2,.25,.05,0.,0.,0.]
    cfg.data.prefiltering = False; cfg.data.preload_dataset = False; cfg.data.visualize_batch = False
    native = ObjectPickDataset(**PickDataset.from_config(cfg.data),split='train')
    # No persistent denylist or implicit single-object repetition may change the selected split.
    native.scenes = [r['uid'] for r in rows]
    native.save_to_denylist = lambda path,key,error: None
    native.grasp_dataset_reader = GraspLabelReader(config,rows)
    return native


def signature(config,rows):
    values = config.to_dict()
    values['trainer'] = trainer(config)
    for key in ('checkpoint','epochs','train_checkpoint_mode','timeout_minutes','gpu'):
        values.pop(key,None)
    files = sorted({row[key] for row in rows for key in ('label','mesh')})
    pins = json.loads((ROOT/'grasppanda/resources/upstreams.lock.json').read_text())
    pin = next(row['pinned_commit'] for row in pins if row['id']=='graspgen')
    return {'config':values,'objects':[r['uid'] for r in rows],'source_commit':pin,'adapter_version':1,
            'files':[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in files]}


def train(config,out,short=False):
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    sampler,cfg,payload = build(config)
    rows = selection(config,inventory(config))
    native = native_dataset(config,cfg,out,rows)
    stage = trainer(config)['stage']
    model = getattr(sampler.model,'grasp_'+stage)
    from grasp_gen.utils.train_utils import build_optimizer
    cfg.optimizer.lr = config.learning_rate
    cfg.train.num_gpus = 1
    optimizer = build_optimizer(cfg,model)
    expected = signature(config,rows)
    start_epoch,steps = 0,0
    resume = config.train_checkpoint_mode == 'resume'
    if resume:
        if not payload.get('resume_supported') or payload.get('signature') != expected or payload.get('trained_stage') != stage:
            raise ValueError('Resume requires an epoch checkpoint with unchanged stage, data and training settings')
        start_epoch,steps = payload['completed_epochs'],payload['completed_steps']
        if config.epochs <= start_epoch: raise ValueError('Set epochs above the completed checkpoint epoch count')
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if not same_state(optimizer.state_dict(),payload['optimizer_state_dict']):
            raise ValueError('GraspGen optimizer state was not restored exactly')
        restore_rng_state(payload['rng'])
    result = {'stage':'short_training' if short else 'epoch_training','dataset':config.dataset,
              'method':config.method,'trained_stage':stage,'training_objects':len(rows),'losses':[],
              'protocol':protocol(config),'resumed':resume,'ap':None,
              'note':'Native stage objective and AdamW parameter groups (weight decay 0.05, native normalization/embedding exceptions). No on-policy generation or simulator evaluation.'}
    epochs = config.training_steps if short else config.epochs
    for epoch in range(start_epoch,epochs):
        model.train()
        order = torch.randperm(len(rows),generator=torch.Generator().manual_seed(config.seed+epoch)).tolist()
        for batch,start in enumerate(range(0,len(order),config.batch_size)):
            if config.train_batch_limit and batch >= config.train_batch_limit: break
            values = []; attempts = []
            for index in order[start:start+config.batch_size]:
                for attempt in range(8):
                    with observation_seed(sample_seed(config,rows[index]['uid'],epoch,attempt)):
                        value = native[index]
                    if not value.get('invalid'): break
                else: raise ValueError('No supervised partial view after 8 attempts: '+rows[index]['uid']+'. Check mesh and labels; no object was silently dropped.')
                values.append(value); attempts.append(attempt)
            data = {'points':torch.stack([torch.as_tensor(v['points']).reshape(-1,3).float() for v in values]).cuda(),
                    'grasps':[torch.as_tensor(v['grasps']).reshape(-1,4,4).float().cuda() for v in values]}
            if stage == 'discriminator':
                data['labels'] = [torch.as_tensor(v['labels']).reshape(-1,1).float().cuda() for v in values]
            if not all(torch.isfinite(t).all() for t in [data['points'],*data['grasps'],*data.get('labels',[])]):
                raise ValueError('Non-finite native training input')
            optimizer.zero_grad(set_to_none=True)
            _,losses,_ = model(data)
            loss = sum(weight*term for weight,term in losses.values())
            if not torch.isfinite(loss): raise ValueError('Non-finite GraspGen loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite GraspGen gradient')
            optimizer.step(); steps += 1
            result['losses'].append({'stage':stage,'epoch':epoch+1,'step':steps,'total':loss.item(),
                'objects':len(values),'render_retries':sum(attempts),**{k:t.item() for k,(_,t) in losses.items()}})
            print(json.dumps(result['losses'][-1]),flush=True); write_result(out,result)
            if short and steps >= config.training_steps: break
        state = {'format':'grasppanda-graspgen-v1','generator':sampler.model.grasp_generator.state_dict(),
                 'discriminator':sampler.model.grasp_discriminator.state_dict(),'trained_stage':stage,
                 'optimizer_state_dict':optimizer.state_dict(),'completed_epochs':epoch+1,'completed_steps':steps,
                 'signature':expected,'rng':capture_rng_state(),'resume_supported':not short}
        temp = out/'checkpoint.pt.tmp'; torch.save(state,temp); temp.replace(out/'checkpoint.pt')
        result.update(checkpoint='checkpoint.pt',completed_epochs=epoch+1,completed_steps=steps)
        write_result(out,result)
        if short and steps >= config.training_steps: break
    return result


def run(config,out):
    prepare()
    if config.action == 'infer': return infer(config,out)
    return train(config,out,short=config.action == 'train_short')
