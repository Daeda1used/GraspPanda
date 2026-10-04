"""Author partial-cloud GraspLDM networks with explicit toolbox rendering and training."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

from ..config import ROOT
from ..integrations.acronym import inventory,selection,object_info,options,trainer,protocol,assets
from ..jobs import digest
from ..training.state import capture_rng_state,restore_rng_state,same_state,write_result


def prepare():
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    from ..runtime.build_graspldm import build
    backend=build()
    module=types.ModuleType('grasp_ldm.models.modules.ext.pvcnn.modules.functional.backend')
    module._backend=backend
    sys.modules[module.__name__]=module
    repo=ROOT/'upstream/object_centric/graspldm'
    sys.path.insert(0,str(repo))
    return repo


def build(config,repo):
    import torch
    from grasp_ldm.models.builder import build_model_from_cfg
    from grasp_ldm.utils.config import Config
    cfg=Config.fromfile(str(repo/'configs/generation/partial_pc/ppc_1a_partial_63cat8k_filtered_latentc3_z16_pc256_180k.py'))
    cfg.model.ddm.model.args.noise_scheduler_type=options(config)['sampler'] if config.action=='infer' else 'ddpm'
    model=build_model_from_cfg(copy.deepcopy(cfg.model.ddm))
    model.set_vae_model(build_model_from_cfg(copy.deepcopy(cfg.model.vae)))
    payload={}
    if config.checkpoint:
        state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
        if state.get('format')=='grasppanda-graspldm-v1':
            payload=state;model.load_state_dict(state['model'],strict=True)
        else:
            state=state['state_dict']
            if not all(k.startswith('model.') for k in state):raise ValueError('Expected the registered GraspLDM online-model checkpoint')
            model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items()},strict=True)
    elif trainer(config)['stage']=='diffusion':
        path=next(iter(assets(config).values()))
        state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
        model.vae_model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items()},strict=True)
    if payload and config.action=='infer' and payload['trained_stage']=='vae' and options(config)['mode']!='vae':
        raise ValueError('A VAE-stage checkpoint requires dataset_options.mode: vae for inference. Train its diffusion stage before diffusion sampling.')
    return model.cuda(),cfg,payload


def render_raw(config,row,view,repo,mesh_path,scale,seed):
    """Generate only a visual observation; grasp labels cannot select the view."""
    import numpy as np
    import pyrender
    import trimesh
    from grasp_ldm.utils.camera import Camera
    mesh=trimesh.load(mesh_path,force='mesh');mesh.apply_scale(scale)
    if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():raise ValueError('Invalid ACRONYM mesh: '+row['uid'])
    center=mesh.vertices.mean(0);mesh.vertices-=center
    rng=np.random.default_rng(seed)
    azimuth=rng.uniform(0,2*np.pi);elevation=rng.uniform(-np.pi/3,np.pi/3)
    distance=rng.uniform(.5,.8)
    position=distance*np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
    z=position/np.linalg.norm(position);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    world_from_gl=np.eye(4);world_from_gl[:3,:3]=np.stack((x,y,z),axis=1);world_from_gl[:3,3]=position
    camera=Camera(str(repo/'grasp_ldm/dataset/cameras/camera_d435i_dummy.json'))
    scene=pyrender.Scene();scene.add(pyrender.Mesh.from_trimesh(mesh));scene.add(camera.to_pyrender_camera(),pose=world_from_gl)
    renderer=pyrender.OffscreenRenderer(camera.width,camera.height)
    try: depth=renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:renderer.delete()
    # Same 0.1 mm depth storage resolution as the author's partial-depth loader.
    if not np.isfinite(depth).all() or np.any(depth<0) or np.any(depth>=6.5536):
        raise ValueError('Rendered depth exceeds the native 16-bit depth range')
    depth=(depth*10000).astype(np.uint16).astype(np.float32)/10000
    points=camera.depth_to_pointcloud(depth).astype(np.float32)
    if len(points)<1024 or not np.isfinite(points).all():raise ValueError('Rendered view has fewer than 1024 valid depth points: '+row['uid'])
    points=points[rng.permutation(len(points))[:1024]]
    world_from_cv=world_from_gl@np.diag([1,-1,-1,1])
    object_to_center=np.eye(4);object_to_center[:3,3]=-center
    camera_from_object=np.linalg.inv(world_from_cv)@object_to_center
    return points,camera_from_object.astype(np.float32)


def render(config,row,view,repo):
    import fcntl
    import numpy as np
    mesh,scale=object_info(config,row)
    seed=int.from_bytes(hashlib.sha256(f'{config.seed}:{row["uid"]}:{view}'.encode()).digest()[:4],'little')
    identity=dict(version=1,mesh_sha256=digest(mesh),scale=scale,seed=seed,
                  camera_sha256=digest(repo/'grasp_ldm/dataset/cameras/camera_d435i_dummy.json'))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=Path(config.label_root) if config.label_root else Path(config.dataset_root)/'.grasppanda/acronym-depth-v1'
    try:cache.mkdir(parents=True,exist_ok=True)
    except OSError as error:raise ValueError('ACRONYM depth cache is not writable; set label_root to a writable storage volume') from error
    path=cache/(key+'.npz');record=cache/(key+'.json')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.is_symlink() or record.is_symlink():raise ValueError('Refusing a symbolic observation cache entry')
        reused=path.is_file() and record.is_file()
        if reused:
            saved=json.loads(record.read_text())
            if saved.get('identity')!=identity or saved.get('sha256')!=digest(path):
                raise ValueError('ACRONYM depth cache changed; move the generated entry aside and retry')
            with np.load(path,allow_pickle=False) as value:points=value['points'];transform=value['object_to_camera']
        else:
            points,transform=render_raw(config,row,view,repo,mesh,scale,seed)
            temporary=path.with_suffix('.tmp')
            if temporary.is_symlink():raise ValueError('Refusing a symbolic partial observation')
            with temporary.open('wb') as stream:np.savez_compressed(stream,points=points,object_to_camera=transform)
            temporary.replace(path)
            temporary=record.with_suffix('.json.tmp')
            if temporary.is_symlink():raise ValueError('Refusing a symbolic partial observation record')
            temporary.write_text(json.dumps(dict(identity=identity,sha256=digest(path)),indent=2)+'\n');temporary.replace(record)
    if points.shape!=(1024,3) or transform.shape!=(4,4) or not np.isfinite(points).all() or not np.isfinite(transform).all():
        raise ValueError('Invalid cached ACRONYM observation')
    return points,transform,dict(seed=seed,mesh_sha256=identity['mesh_sha256'],annotation_sha256=digest(row['path']),
                                 observation_sha256=digest(path),render_cache_reused=reused)


def preview(points,poses,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from grasp_ldm.utils.gripper import SimplePandaGripper
    scene=trimesh.Scene();scene.add_geometry(trimesh.points.PointCloud(points,colors=[113,138,153,255]))
    fig=plt.figure(figsize=(8,6),layout='constrained');ax=fig.add_subplot(projection='3d')
    ax.scatter(*points.T,s=2,c='#718a99',alpha=.6);bounds=[points]
    for pose in poses[:3]:
        mesh=SimplePandaGripper.create_gripper_marker(color=[41,179,145,255]);mesh.apply_transform(pose)
        scene.add_geometry(mesh);bounds.append(mesh.vertices)
        ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces],facecolor='#29b391',alpha=.8))
    points_all=np.concatenate(bounds);center=(points_all.min(0)+points_all.max(0))/2;extent=max(np.ptp(points_all,axis=0).max()*.55,.01)
    ax.set(xlabel='Camera x (m)',ylabel='Camera y (m)',zlabel='Camera z (m)',
           xlim=(center[0]-extent,center[0]+extent),ylim=(center[1]-extent,center[1]+extent),zlim=(center[2]-extent,center[2]+extent),
           title='Partial depth · native GraspLDM poses');ax.set_box_aspect((1,1,1))
    fig.savefig(out/'preview.png',dpi=130);plt.close(fig);scene.export(out/'grasp_scene.glb')


def infer(config,out,repo):
    import numpy as np
    import torch
    from grasp_ldm.utils.rotations import tmrp_to_H
    model,_,_=build(config,repo);model.eval();settings=options(config)
    model.set_inference_timesteps(settings['inference_steps'])
    output=out/'predictions';output.mkdir(exist_ok=True);records=[]
    for row,view in selection(config,inventory(config)):
        points,transform,source=render(config,row,view,repo)
        mean=points.mean(0);pc=torch.from_numpy((points-mean)/.05)[None].cuda()
        torch.cuda.synchronize();started=time.monotonic()
        with torch.inference_mode():
            if settings['mode']=='vae':value=model.vae_model.generate_grasps(pc,settings['num_grasps'])
            else:value=model.generate_grasps(pc,settings['num_grasps'])[0]
            pose=value[0]*torch.tensor([.05,.05,.05,.5,.5,.5],device='cuda')
            pose[:,:3]+=torch.from_numpy(mean).cuda()
            poses=tmrp_to_H(pose).cpu().numpy();scores=value[1].sigmoid().flatten().cpu().numpy()
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        if poses.shape!=(settings['num_grasps'],4,4) or scores.shape!=(len(poses),) or not np.isfinite(poses).all() or not np.isfinite(scores).all():
            raise ValueError('Invalid native GraspLDM output')
        if not np.allclose(poses[:,:3,:3]@poses[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-4):
            raise ValueError('Invalid GraspLDM rotations')
        order=np.argsort(scores)[::-1];poses,scores=poses[order],scores[order]
        name=f'{row["uid"]}_view{view:02d}.npz'
        np.savez_compressed(output/name,poses=poses,scores=scores,points=points,object_to_camera=transform)
        records.append(dict(object=row['uid'],view=view,prediction=name,grasps=len(poses),sha256=digest(output/name),
                            forward_decode_seconds=elapsed,**source))
        if len(records)==1:preview(points,poses,out)
    result=dict(stage='dataset_inference',dataset=config.dataset,method=config.method,frames=records,prediction_dir='predictions',
                protocol=protocol(config),checkpoint_sha256=digest(config.checkpoint),ap=None,
                note='Native partial-cloud model and decoding; toolbox visual rendering. Confidence is not calibrated simulation success.')
    (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def sample(config,row,view,repo,preprocessor):
    import h5py
    import numpy as np
    import torch
    from grasp_ldm.utils.rotations import H_to_tmrp
    points,transform,_=render(config,row,view,repo)
    with h5py.File(row['path'],'r') as source:
        if isinstance(source.get('grasps',getlink=True),h5py.ExternalLink):raise ValueError('External grasp-label links are not supported')
        poses=np.asarray(source['grasps/transforms'],dtype=np.float32)
        labels=np.asarray(source['grasps/qualities/flex/object_in_gripper'])
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or labels.shape!=(len(poses),) or not np.isfinite(poses).all():
        raise ValueError('Invalid ACRONYM grasp labels: '+row['uid'])
    poses=poses[labels==1]
    if not len(poses):raise ValueError('No successful labelled grasps for '+row['uid'])
    count=trainer(config)['grasps_per_view']
    ids=np.random.choice(len(poses),count,replace=len(poses)<count)
    transformed=torch.from_numpy(transform@poses[ids])
    grasps=torch.cat((H_to_tmrp(transformed),torch.ones(count,1)),dim=-1)
    pc,grasps,_=preprocessor.preprocess_data(torch.from_numpy(points),grasps)
    return pc,grasps,len(poses)


def signature(config,rows):
    from ..runtime.build_graspldm import COMMIT
    values=config.to_dict();values['trainer']=trainer(config);values['dataset_options']=options(config)
    for name in ('checkpoint','epochs','train_checkpoint_mode','timeout_minutes','gpu'):values.pop(name,None)
    identities={}
    for row,_ in rows:
        if row['uid'] not in identities:
            mesh,scale=object_info(config,row)
            identities[row['uid']]=dict(labels=digest(row['path']),mesh=digest(mesh),scale=scale)
    return dict(config=values,source=COMMIT,adapter_version=1,inputs=identities,views=[[row['uid'],view] for row,view in rows])


def train(config,out,repo,short=False):
    import numpy as np
    import torch
    from grasp_ldm.dataset.acronym.acronym_partial_pointclouds import AcronymPartialPointclouds
    from grasp_ldm.dataset.augmentations import Augmentations
    model,cfg,payload=build(config,repo);stage=trainer(config)['stage'];rows=selection(config,inventory(config))
    network=model.vae_model if stage=='vae' else model
    if stage=='diffusion':model.freeze_vae_model()
    else:model.vae_model.requires_grad_(True)
    optimizer=torch.optim.Adam((p for p in network.parameters() if p.requires_grad),lr=config.learning_rate)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60000,120000],gamma=.1)
    expected=signature(config,rows);start_epoch=steps=0
    if config.train_checkpoint_mode=='resume':
        if not payload.get('resume_supported') or payload.get('signature')!=expected:raise ValueError('Resume requires an epoch checkpoint with unchanged data, stage and training settings')
        start_epoch,steps=payload['completed_epochs'],payload['completed_steps']
        if config.epochs<=start_epoch:raise ValueError('Increase epochs beyond the completed checkpoint epoch count')
        optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
        if not same_state(optimizer.state_dict(),payload['optimizer']):raise ValueError('GraspLDM optimizer restoration mismatch')
        restore_rng_state(payload['rng'])
    preprocessor=AcronymPartialPointclouds.__new__(AcronymPartialPointclouds)
    preprocessor._set_normalization_params(False)
    preprocessor.augmentations=Augmentations.build_augmentations_from_cfg(cfg.augs_config)
    result=dict(stage='short_training' if short else 'epoch_training',training_stage=stage,dataset=config.dataset,method=config.method,
                selected_views=len(rows),objects=len({r['uid'] for r,_ in rows}),losses=[],validation=[],ap=None,
                resumed=config.train_checkpoint_mode=='resume',protocol=protocol(config),
                note='Native objectives and Adam with toolbox all-success supervision; original visibility-filtered training is not reproduced. No benchmark metric.')
    for epoch in range(start_epoch,config.training_steps if short else config.epochs):
        network.train()
        if stage=='diffusion':model.vae_model.eval()
        order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(config.seed+epoch)).tolist()
        for batch,start in enumerate(range(0,len(order),config.batch_size)):
            if config.train_batch_limit and batch>=config.train_batch_limit:break
            items=[sample(config,*rows[i],repo,preprocessor) for i in order[start:start+config.batch_size]]
            pc=torch.stack([r[0] for r in items]).cuda();grasps=torch.cat([r[1] for r in items]).cuda()
            if stage=='vae':network.latent_loss.set_weight_from_schedule(steps)
            optimizer.zero_grad(set_to_none=True);_,loss=network(pc,grasps)
            if not torch.isfinite(loss.loss):raise ValueError('Non-finite GraspLDM training loss')
            loss.loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):raise ValueError('Non-finite GraspLDM gradient')
            optimizer.step();scheduler.step();steps+=1
            row=dict(epoch=epoch+1,step=steps,stage=stage,total=loss.loss.item(),views=len(items),grasp_targets=len(grasps),
                     **{k:v.item() for k,v in loss.items() if k!='loss'})
            result['losses'].append(row);print(json.dumps(row),flush=True);write_result(out,result)
            if short and steps>=config.training_steps:break
        state=dict(format='grasppanda-graspldm-v1',model=model.state_dict(),trained_stage=stage,optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),rng=capture_rng_state(),signature=expected,completed_epochs=epoch+1,
                   completed_steps=steps,resume_supported=not short)
        temporary=out/'checkpoint.pt.tmp';torch.save(state,temporary);temporary.replace(out/'checkpoint.pt')
        result.update(checkpoint='checkpoint.pt',completed_epochs=epoch+1,completed_steps=steps);write_result(out,result)
        if short and steps>=config.training_steps:break
    return result


def run(config,out):
    repo=prepare()
    return infer(config,out,repo) if config.action=='infer' else train(config,out,repo,config.action=='train_short')
