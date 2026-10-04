"""Native VGN generation, voxel prediction, finite training and physics evaluation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

from ..integrations.vgn import ASSET_SHA,options,inventory,selection,protocol,generation_plan
from ..jobs import digest
from ..training.state import capture_rng_state,restore_rng_state,same_state,write_result


def prepare():
    from ..runtime.vgn_source import source
    repo=source();sys.path[:0]=[str(repo/'src'),str(repo/'scripts')]
    return repo


def assets(config):
    root=Path(config.dataset_root);archive=root/'assets/vgn-data.zip'
    if digest(archive)!=ASSET_SHA:raise ValueError('VGN asset archive differs from the registered download')
    print('Verifying native VGN object assets',flush=True)
    parent=root/'.grasppanda/assets';parent.mkdir(parents=True,exist_ok=True);target=parent/ASSET_SHA[:16]
    with (parent/'extract.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=parent) as temporary:
                work=Path(temporary)/'assets';work.mkdir();files={}
                with zipfile.ZipFile(archive) as z:
                    for index,member in enumerate(z.infolist()):
                        if member.is_dir():continue
                        path=work/member.filename
                        if not path.resolve().is_relative_to(work.resolve()) or (member.external_attr>>16)&0o170000 not in (0,0o100000):
                            raise ValueError('Unsafe VGN archive entry')
                        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(member));files[member.filename]=digest(path)
                        if (index+1)%200==0:print(f'Extracted {index+1}/{len(z.infolist())} asset entries',flush=True)
                (work/'.complete').write_text(json.dumps(files));work.rename(target)
        if not (target/'.complete').is_file():raise ValueError('Incomplete VGN asset cache; move it aside and retry')
        for name,sha in json.loads((target/'.complete').read_text()).items():
            path=target/name
            if path.is_symlink() or not path.is_file() or digest(path)!=sha:raise ValueError('VGN asset cache changed; move it aside and retry')
    print('Native VGN assets verified',flush=True)
    os.chdir(target)
    return target


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    if path.is_symlink() or temporary.is_symlink():raise ValueError('Refusing a symbolic VGN output')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def generate(config,out):
    import numpy as np
    import open3d as o3d
    import pandas as pd
    import generate_data as native
    from vgn.io import write_setup
    from vgn.perception import create_tsdf
    from vgn.simulation import ClutterRemovalSim
    root=Path(config.dataset_root);settings=options(config);assets(config)
    plan=generation_plan(config)
    with (root/'.generate.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('A generation job already owns this dataset root') from None
        manifest=root/'generation.json'
        if manifest.exists() and json.loads(manifest.read_text())!=plan:
            raise ValueError('This root has another generation plan; use a new dataset root to change its settings')
        if not manifest.exists():
            if any((root/p).exists() for p in ('raw','processed','splits.json')):raise ValueError('Refusing to overwrite an existing dataset without a matching generation plan')
            write_json(manifest,plan)
        for directory in ('raw/scenes','raw/labels','processed/scenes','processed/records'):(root/directory).mkdir(parents=True,exist_ok=True)
        records=[]
        for index in range(settings['scene_count']):
            name=f'scene_{index:08d}';record=root/'processed/records'/(name+'.json')
            if record.is_file():
                saved=json.loads(record.read_text())
                for relative,sha in saved['files'].items():
                    path=root/relative
                    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file() or digest(path)!=sha:
                        raise ValueError('Generated VGN scene changed: '+name)
                records.append(saved);print('Reused '+name,flush=True);continue
            seed=int.from_bytes(hashlib.sha256(f'{config.seed}:{index}'.encode()).digest()[:4],'little')
            np.random.seed(seed)
            sim=ClutterRemovalSim(settings['scene_type'],settings['scene_type']+'/train',gui=False,seed=seed)
            try:
                # Native filesystem discovery is sorted for repeatable object sampling.
                sim.object_urdfs.sort();sim.reset(np.random.poisson(native.OBJECT_COUNT_LAMBDA)+1);sim.save_state()
                count=np.random.randint(settings['max_views'])+1
                depth,extrinsics=native.render_images(sim,count)
                surface=create_tsdf(sim.size,120,depth,sim.camera.intrinsic,extrinsics).get_cloud()
                surface=surface.crop(o3d.geometry.AxisAlignedBoundingBox(sim.lower,sim.upper))
                if surface.is_empty() or not (np.asarray(surface.normals)[:,2]>-.1).any():
                    raise ValueError('No native grasp sampling surface for '+name+'; choose another seed and dataset root')
                write_setup(root/'raw',sim.size,sim.camera.intrinsic,sim.gripper.max_opening_width,sim.gripper.finger_depth)
                raw=root/'raw/scenes'/(name+'.npz');grid_path=root/'processed/scenes'/(name+'.npz')
                np.savez_compressed(raw,depth_imgs=depth,extrinsics=extrinsics)
                grid=create_tsdf(sim.size,40,depth,sim.camera.intrinsic,extrinsics).get_grid()
                np.savez_compressed(grid_path,grid=grid,points=np.asarray(surface.points,dtype=np.float32))
                rows=[]
                for trial in range(settings['grasps_per_scene']):
                    point,normal=native.sample_grasp_point(surface,sim.gripper.finger_depth)
                    grasp,label=native.evaluate_grasp_point(sim,point,normal)
                    rows.append([name,*grasp.pose.rotation.as_quat(),*grasp.pose.translation,grasp.width,label])
                    if (trial+1)%20==0:print(f'{name}: {trial+1}/{settings["grasps_per_scene"]} native candidate evaluations',flush=True)
                table=pd.DataFrame(rows,columns=['scene_id','qx','qy','qz','qw','x','y','z','width','label'])
                label_file=root/'raw/labels'/(name+'.csv');table.to_csv(label_file,index=False)
                saved=dict(scene=name,seed=seed,views=count,trials=len(table),successful=int(table.label.sum()),
                           files={str(p.relative_to(root)):digest(p) for p in (raw,grid_path,label_file)})
                write_json(record,saved);records.append(saved)
                print(json.dumps({k:v for k,v in saved.items() if k!='files'}),flush=True)
                write_result(out,dict(stage='dataset_generation',completed_scenes=len(records),requested_scenes=settings['scene_count'],protocol=protocol(config)))
            finally:sim.world.close()
        table=pd.concat([pd.read_csv(root/'raw/labels'/(r['scene']+'.csv')) for r in records],ignore_index=True)
        table.to_csv(root/'raw/grasps.csv',index=False)
        # Native notebook workspace cleaning, followed by native voxel-coordinate conversion.
        kept=table[table[['x','y','z']].ge(.02).all(axis=1)&table[['x','y','z']].le(.28).all(axis=1)].copy()
        if set(kept.scene_id)!=set(r['scene'] for r in records):raise ValueError('A generated scene has no labels within the native training bounds; use more candidates or another dataset root')
        for field in ('x','y','z','width'):kept[field]/=.3/40
        kept.rename(columns=dict(x='i',y='j',z='k')).to_csv(root/'processed/grasps.csv',index=False)
        names=sorted(r['scene'] for r in records);order=np.random.default_rng(settings['split_seed']).permutation(len(names))
        cut=min(max(int(len(names)*settings['train_fraction']),1),len(names)-1)
        write_json(root/'splits.json',dict(train=sorted(names[i] for i in order[:cut]),val=sorted(names[i] for i in order[cut:])))
    return dict(stage='dataset_generation',scenes=len(records),raw_trials=len(table),cleaned_trials=len(kept),
                positive_trials=int(kept.label.sum()),dataset_root=str(root),records=records,protocol=protocol(config),
                note='Native simulation labels; toolbox deterministic scene orchestration and scene-disjoint splits. No fixed benchmark was downloaded.')


def build(config):
    import torch
    from vgn.networks import get_network
    net=get_network('conv');payload={}
    if config.checkpoint:
        value=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
        if value.get('format')=='grasppanda-vgn-v1':payload=value;value=value['model']
        net.load_state_dict(value,strict=True)
    return net.cuda(),payload


def predict(config,net,grid,voxel_size):
    import numpy as np
    from vgn.detection import predict as forward,process,select
    from vgn.grasp import from_voxel_coordinates
    if grid.shape!=(1,40,40,40) or not np.isfinite(grid).all():raise ValueError('Invalid native VGN TSDF')
    net.eval();quality,rotation,width=process(grid,*forward(grid,net,'cuda'))
    if not all(np.isfinite(a).all() for a in (quality,rotation,width)):raise ValueError('Non-finite VGN prediction')
    grasps,scores=select(quality,rotation,width,threshold=options(config)['quality_threshold'])
    order=np.random.permutation(len(grasps))
    return [from_voxel_coordinates(grasps[i],voxel_size) for i in order],np.asarray(scores)[order]


def preview(points,grasps,out):
    import numpy as np
    import trimesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    scene=trimesh.Scene();scene.add_geometry(trimesh.points.PointCloud(points,colors=[113,138,153,255]))
    fig=plt.figure(figsize=(8,6),layout='constrained');ax=fig.add_subplot(projection='3d');ax.scatter(*points.T,s=1,c='#718a99',alpha=.5)
    for grasp in grasps[:3]:
        width=grasp.width;depth=.05
        segments=[[[0,0,-depth/2],[0,0,0]],[[0,-width/2,0],[0,-width/2,depth]],[[0,width/2,0],[0,width/2,depth]],[[0,-width/2,0],[0,width/2,0]]]
        for segment in segments:
            mesh=trimesh.creation.cylinder(radius=.0015,segment=np.asarray(segment));mesh.apply_transform(grasp.pose.as_matrix());mesh.visual.vertex_colors=[41,179,145,255]
            scene.add_geometry(mesh);ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces],facecolor='#29b391',alpha=.8))
    ax.set(xlabel='Workspace x (m)',ylabel='Workspace y (m)',zlabel='Workspace z (m)',xlim=(0,.3),ylim=(0,.3),zlim=(0,.3),title='Depth TSDF · native VGN grasps');ax.set_box_aspect((1,1,1))
    fig.savefig(out/'preview.png',dpi=130);plt.close(fig);scene.export(out/'grasp_scene.glb')


def infer(config,out):
    import numpy as np
    net,_=build(config);root=Path(config.dataset_root);target=out/'predictions';target.mkdir(exist_ok=True);records=[]
    size=json.loads((root/'raw/setup.json').read_text())['size']
    for name in selection(config,inventory(config)):
        path=root/'processed/scenes'/(name+'.npz')
        with np.load(path,allow_pickle=False) as data:grid=data['grid'];points=data['points']
        start=time.monotonic();grasps,scores=predict(config,net,grid,size/40)
        poses=np.asarray([g.pose.as_matrix() for g in grasps],dtype=np.float32).reshape(-1,4,4)
        output=target/(name+'.npz');np.savez_compressed(output,poses=poses,widths=np.asarray([g.width for g in grasps]),scores=scores,points=points)
        records.append(dict(scene=name,grasps=len(grasps),prediction=output.name,sha256=digest(output),observation_sha256=digest(path),seconds=time.monotonic()-start))
        if len(records)==1:preview(points,grasps,out)
    result=dict(stage='dataset_inference',method=config.method,dataset=config.dataset,frames=records,prediction_dir='predictions',protocol=protocol(config),ap=None)
    write_json(target/'manifest.json',result);return result


def train(config,out):
    import numpy as np
    import pandas as pd
    import torch
    from vgn.dataset import Dataset
    import train_vgn as native
    from ..runtime.vgn_source import COMMIT
    root=Path(config.dataset_root);net,payload=build(config);settings=options(config);short=config.action=='train_short'
    all_rows=pd.read_csv(root/'processed/grasps.csv');names=selection(config,inventory(config))
    def loader(split,selected):
        table=all_rows[all_rows.scene_id.isin(selected)].copy()
        # Balance within each split, never before assigning complete scenes.
        positive=table[table.label==1];negative=table[table.label==0];count=min(len(positive),len(negative))
        if not count:raise ValueError('VGN '+split+' needs both positive and negative trials; generate more scenes/candidates')
        table=pd.concat([positive.sample(count,random_state=settings['split_seed']),negative.sample(count,random_state=settings['split_seed'])]).reset_index(drop=True)
        dataset=Dataset(root/'processed',augment=settings['augment'] and split=='train');dataset.df=table
        return dataset
    training=loader('train',names);validation=loader('val',inventory(config,'val')) if not short else None
    cfg=config.to_dict()
    for key in ('epochs','checkpoint','train_checkpoint_mode','timeout_minutes','gpu'):cfg.pop(key,None)
    signature=dict(config=cfg,source=COMMIT,adapter=1,labels=digest(root/'processed/grasps.csv'),splits=digest(root/'splits.json'),
                   observations={name:digest(root/'processed/scenes'/(name+'.npz')) for name in sorted(set(names+(inventory(config,'val') if validation else [])))})
    optimizer=torch.optim.Adam(net.parameters(),lr=config.learning_rate);start_epoch=steps=0
    if config.train_checkpoint_mode=='resume':
        if not payload.get('resume_supported') or payload.get('signature')!=signature:raise ValueError('Resume requires unchanged VGN data and training settings')
        start_epoch,steps=payload['completed_epochs'],payload['completed_steps']
        if config.epochs<=start_epoch:raise ValueError('Increase epochs beyond the completed checkpoint epoch count')
        optimizer.load_state_dict(payload['optimizer'])
        if not same_state(optimizer.state_dict(),payload['optimizer']):raise ValueError('VGN optimizer restoration mismatch')
        restore_rng_state(payload['rng'])
    result=dict(stage='short_training' if short else 'epoch_training',method=config.method,dataset=config.dataset,losses=[],validation=[],
                training_labels=len(training),validation_labels=len(validation) if validation else 0,protocol=protocol(config),ap=None)
    for epoch in range(start_epoch,config.training_steps if short else config.epochs):
        loader_train=torch.utils.data.DataLoader(training,batch_size=config.batch_size,shuffle=True,num_workers=0,
                                                 generator=torch.Generator().manual_seed(config.seed+epoch))
        net.train()
        for batch,value in enumerate(loader_train):
            if config.train_batch_limit and batch>=config.train_batch_limit:break
            x,y,index=native.prepare_batch(value,'cuda');optimizer.zero_grad(set_to_none=True)
            loss=native.loss_fn(native.select(net(x),index),y)
            if not torch.isfinite(loss):raise ValueError('Non-finite VGN loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('Non-finite VGN gradient')
            optimizer.step();steps+=1
            row=dict(epoch=epoch+1,step=steps,total=loss.item(),labels=len(x));result['losses'].append(row);print(json.dumps(row),flush=True);write_result(out,result)
            if short and steps>=config.training_steps:break
        if validation:
            net.eval();total=correct=seen=0
            with torch.inference_mode():
                for batch,value in enumerate(torch.utils.data.DataLoader(validation,batch_size=config.batch_size,num_workers=0)):
                    if config.eval_batch_limit and batch>=config.eval_batch_limit:break
                    x,y,index=native.prepare_batch(value,'cuda');pred=native.select(net(x),index);loss=native.loss_fn(pred,y)
                    if not torch.isfinite(loss):raise ValueError('Non-finite VGN validation loss')
                    total+=loss.item()*len(x);correct+=int(((pred[0]>=.5)==y[0]).sum());seen+=len(x)
            result['validation'].append(dict(epoch=epoch+1,labels=seen,total=total/seen,label_accuracy=correct/seen))
        state=dict(format='grasppanda-vgn-v1',model=net.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng_state(),signature=signature,
                   completed_epochs=epoch+1,completed_steps=steps,resume_supported=not short)
        temporary=out/'checkpoint.pt.tmp';torch.save(state,temporary);temporary.replace(out/'checkpoint.pt')
        result.update(checkpoint='checkpoint.pt',completed_epochs=epoch+1,completed_steps=steps);write_result(out,result)
        if short and steps>=config.training_steps:break
    return result


def simulate(config,out):
    import numpy as np
    import pandas as pd
    from vgn.experiments import clutter_removal as native
    assets(config);settings=options(config);net,_=build(config);seen=0
    def planner(state):
        nonlocal seen
        start=time.monotonic();grasps,scores=predict(config,net,state.tsdf.get_grid(),state.tsdf.voxel_size)
        elapsed=time.monotonic()-start
        if seen==0:preview(np.asarray(state.pc.points),grasps,out)
        seen+=1;return grasps,scores,elapsed
    native.run(planner,out/'simulation','native_vgn',settings['scene_type'],settings['scene_type']+'/test',
               num_objects=settings['simulation_objects'],n=settings['simulation_views'],num_rounds=settings['simulation_rounds'],seed=config.seed,sim_gui=False,rviz=False)
    directory=next((out/'simulation').iterdir());rounds=pd.read_csv(directory/'rounds.csv');grasps=pd.read_csv(directory/'grasps.csv')
    successful=int(grasps.label.sum());objects=int(rounds.object_count.sum());attempts=len(grasps)
    return dict(stage='simulation',method=config.method,dataset=config.dataset,rounds=len(rounds),objects=objects,attempts=attempts,successful_grasps=successful,
                metrics=dict(simulated_grasp_success=successful/attempts if attempts else None,simulated_objects_cleared=successful/objects if objects else None),
                protocol=protocol(config),logs=str(directory.relative_to(out)),ap=None,
                note='Native closed-loop clutter removal on author test object assets. Simulated success, not real-robot or full-paper reproduction.')


def run(config,out):
    prepare()
    if config.action=='generate':return generate(config,out)
    if config.action=='simulate':return simulate(config,out)
    if config.action=='infer':return infer(config,out)
    return train(config,out)
