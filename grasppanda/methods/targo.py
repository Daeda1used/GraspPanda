"""Native target-conditioned TARGO prediction and finite grasp training."""
import json
import os
from pathlib import Path
import sys
import time

from ..config import ROOT
from ..integrations.targo import inventory, selection, options, protocol, assets
from ..jobs import digest
from ..training.state import capture_rng_state, restore_rng_state, same_state, write_result


def prepare():
    from ..runtime.build_targo import build
    repo = build()
    sys.path[:0] = [str(repo),str(repo/'src'),str(repo/'src/shape_completion/chamfer_dist')]
    os.chdir(repo)
    # Released visual arrays are numeric; loading them must not execute pickle payloads.
    import numpy as np
    from vgn import io
    class NumericNumpy:
        def __getattr__(self,name): return getattr(np,name)
        def load(self,*args,**kwargs):
            return np.load(*args,**{**kwargs,'allow_pickle':False})
    io.np = NumericNumpy()
    return repo


def build(config,repo,completion=False):
    import torch
    from vgn.networks import get_network
    net = get_network('targo').cuda()
    payload = {}
    if config.checkpoint:
        value = torch.load(config.checkpoint,map_location='cpu',weights_only=True)
        if value.get('format') == 'grasppanda-targo-v1':
            payload = value; value = value['model']
        net.load_state_dict(value,strict=True)
    net.eval()
    sc = None
    if completion:
        from shape_completion.models.AdaPoinTr import AdaPoinTr
        from shape_completion.config import cfg_from_yaml_file
        cfg = cfg_from_yaml_file(str(repo/'src/shape_completion/configs/AdaPoinTr.yaml'))
        sc = AdaPoinTr(cfg.model).cuda().eval()
        path = next(iter(assets(config).values()))
        value = torch.load(path,map_location='cpu',weights_only=True)['base_model']
        sc.load_state_dict({k.removeprefix('module.'):v for k,v in value.items()},strict=True)
        sc.requires_grad_(False)
    return net,sc,payload


def observation(row,repo):
    import numpy as np
    import torch
    from utils_giga import filter_and_pad_point_clouds
    with np.load(row['path'],allow_pickle=False) as data:
        target = data['pc_depth_targ'].astype(np.float32)
        background = data['pc_scene_no_targ'].astype(np.float32)
    for cloud in (target,background):
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError('Invalid TARGO visual points in '+row['uid'])
    visible = np.all((target >= [.02,.02,.055]) & (target <= [.28,.28,.3]),axis=1)
    if not visible.any(): raise ValueError('No observed target points inside the TARGO workspace: '+row['uid'])
    target = filter_and_pad_point_clouds(torch.from_numpy(target/.3-.5)[None])[0].numpy()
    plane = np.load(repo/'setup/plane_sampled.npy',allow_pickle=False).astype(np.float32)
    return np.concatenate((background,plane))/.3-.5,target


def preview(scene_points,target,complete,grasps,scores,out):
    import numpy as np
    import trimesh
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from vgn.utils.visual import grasp2mesh
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(scene_points,colors=[115,138,153,255]),node_name='observed_scene')
    scene.add_geometry(trimesh.points.PointCloud(target,colors=[224,135,83,255]),node_name='observed_target')
    fig = plt.figure(figsize=(8,6),layout='constrained'); ax = fig.add_subplot(projection='3d')
    ax.scatter(*scene_points.T,s=1,c='#7893a1',alpha=.35)
    ax.scatter(*target.T,s=2,c='#df8753'); ax.scatter(*complete.T,s=1,c='#35b49c',alpha=.2)
    for index,(grasp,score) in enumerate(zip(grasps[:3],scores[:3])):
        mesh = grasp2mesh(grasp,float(score))
        scene.add_geometry(mesh,node_name=f'predicted_gripper_{index}')
        ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces],facecolor='#25aa87',alpha=.65))
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',xlim=(0,.3),ylim=(0,.3),zlim=(0,.3),
           title='Target-conditioned depth · native TARGO grasps')
    ax.set_box_aspect((1,1,1)); fig.savefig(out/'preview.png',dpi=130); plt.close(fig)
    scene.export(out/'grasp_scene.glb')


def infer(config,out,repo):
    import numpy as np
    import torch
    from vgn.detection_implicit_vgn import predict_targo,process,bound,select
    from utils_giga import point_cloud_to_tsdf
    net,sc,_ = build(config,repo,completion=True)
    settings = options(config)
    target_dir = out/'predictions'; target_dir.mkdir(exist_ok=True)
    records = []
    for row in selection(config,inventory(config)):
        scene,target = observation(row,repo)
        torch.cuda.synchronize(); started = time.monotonic()
        with torch.inference_mode():
            completed = sc(torch.from_numpy(target)[None].cuda())[1][0].cpu().numpy()
        if completed.shape != (2048,3) or not np.isfinite(completed).all():
            raise ValueError('Invalid AdaPoinTr completion; no partial-target fallback is applied')
        completed = np.clip(completed,-.5,.5-1e-6)
        grid = point_cloud_to_tsdf((completed+.5)*.3)
        qual,rot,width = predict_targo((scene,completed),net,torch.device('cuda'))
        if not all(np.isfinite(x).all() for x in (grid,qual,rot,width)):
            raise ValueError('Non-finite TARGO prediction')
        qual,rot,width = process(grid,qual.reshape(40,40,40),rot.reshape(40,40,40,4),width.reshape(40,40,40),
                                 out_th=settings['outside_threshold'])
        qual = bound(qual,.0075)
        pos = torch.stack(torch.meshgrid(*[torch.linspace(-.5,.475,40)]*3,indexing='ij'),dim=-1)
        grasps,scores = select(qual,pos,rot,width,threshold=settings['quality_threshold'],force_detection=settings['force_detection'])
        order = np.argsort(scores)[::-1]
        grasps,scores = [grasps[i] for i in order],np.asarray(scores,dtype=np.float32)[order]
        for grasp in grasps:
            grasp.pose.translation = (grasp.pose.translation+.5)*.3
            grasp.width *= .3
        poses = np.asarray([g.pose.as_matrix() for g in grasps],dtype=np.float32).reshape(-1,4,4)
        widths = np.asarray([g.width for g in grasps],dtype=np.float32)
        torch.cuda.synchronize(); elapsed = time.monotonic()-started
        name = row['uid']+'.npz'
        np.savez_compressed(target_dir/name,poses=poses,widths=widths,scores=scores,
                            scene_points=(scene+.5)*.3,target_points=(target+.5)*.3,completed_target=(completed+.5)*.3)
        records.append(dict(scene=row['uid'],prediction=name,sha256=digest(target_dir/name),
                            observation_sha256=digest(row['path']),grasps=len(grasps),
                            forced_below_threshold=bool(len(scores) and scores[0]<settings['quality_threshold']),
                            completion_decode_seconds=elapsed))
        if len(records)==1: preview((scene+.5)*.3,(target+.5)*.3,(completed+.5)*.3,grasps,scores,out)
    result = dict(stage='dataset_inference',dataset=config.dataset,method=config.method,frames=records,
                  prediction_dir='predictions',protocol=protocol(config),checkpoint_sha256=digest(config.checkpoint),ap=None,
                  note='Native confidence-ranked target grasps. Shape completion is required; simulator success is not measured.')
    (target_dir/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def native_dataset(config,rows):
    import torch
    from vgn.dataset_voxel import DatasetVoxel_Target
    from vgn import io
    root = Path(config.dataset_root)/'syn_train'
    native = DatasetVoxel_Target(root,root,augment=False,ablation_dataset='',model_type='targo',
                                data_contain='pc and targ_grid',shape_completion=True)
    ids = {row['uid'] for row in rows}
    native.df = native.df[native.df.scene_id.isin(ids)].reset_index(drop=True)
    if not len(native.df): raise ValueError('No labelled grasps in the selected TARGO scenes')
    io._SCENE_INDEX_CACHE.setdefault((str(root),'scenes'),{}).update({row['uid']:row['path'] for row in rows})
    class StrictDataset(torch.utils.data.Dataset):
        def __len__(self): return len(native)
        def __getitem__(self,index):
            uid = native.df.loc[index,'scene_id']
            # Bypass the author retry-next-row wrapper: each selected label must be consumed as selected.
            try: return native._get_item_safe(index,uid)
            except Exception as error: raise ValueError('Unable to load TARGO label for scene '+uid+': '+str(error)) from error
    return StrictDataset()


def signature(config,rows,validation):
    values = config.to_dict(); values['dataset_options'] = options(config)
    for key in ('checkpoint','epochs','train_checkpoint_mode','timeout_minutes','gpu'): values.pop(key,None)
    files = sorted({r['path'] for r in rows+validation})
    from ..runtime.build_targo import COMMIT
    return dict(config=values,source=COMMIT,adapter_version=1,objects=[r['uid'] for r in rows],
                validation=[r['uid'] for r in validation],
                labels_sha256=digest(Path(config.dataset_root)/'syn_train/grasps.csv'),
                files=[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in files])


def train(config,out,repo,short=False):
    import numpy as np
    import torch
    import train_targo as native
    net,_,payload = build(config,repo)
    rows = selection(config,inventory(config))
    validation = [] if short else inventory(config,'val')
    data = native_dataset(config,rows)
    val_data = native_dataset(config,validation) if validation else None
    optimizer = torch.optim.Adam(net.parameters(),lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.95)
    expected = signature(config,rows,validation)
    start_epoch,steps = 0,0
    resumed = config.train_checkpoint_mode=='resume'
    if resumed:
        if not payload.get('resume_supported') or payload.get('signature')!=expected:
            raise ValueError('TARGO resume requires an epoch checkpoint with unchanged data and training settings')
        start_epoch,steps = payload['completed_epochs'],payload['completed_steps']
        if config.epochs<=start_epoch: raise ValueError('Increase epochs above the completed checkpoint epoch count')
        optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
        if not same_state(optimizer.state_dict(),payload['optimizer']): raise ValueError('TARGO optimizer state was not restored exactly')
        restore_rng_state(payload['rng'])
    result = dict(stage='short_training' if short else 'epoch_training',dataset=config.dataset,method=config.method,
                  training_scenes=len(rows),training_labels=len(data),validation_scenes=len(validation),
                  validation_labels=len(val_data) if val_data else 0,losses=[],validation=[],resumed=resumed,
                  protocol=protocol(config),ap=None,
                  note='Native grasp objective and Adam; ExponentialLR gamma 0.95 every 10 epochs. Frozen AdaPoinTr is used at inference. Validation loss is not simulator success.')
    for epoch in range(start_epoch,config.training_steps if short else config.epochs):
        loader = torch.utils.data.DataLoader(data,batch_size=config.batch_size,shuffle=True,num_workers=0,
                    generator=torch.Generator().manual_seed(config.seed+epoch),drop_last=False)
        net.train()
        for batch,value in enumerate(loader):
            if config.train_batch_limit and batch>=config.train_batch_limit: break
            x,y,pos = native.prepare_batch(value,torch.device('cuda'))
            optimizer.zero_grad(set_to_none=True)
            loss,parts = native.loss_fn(native.select(net(x,pos)),y)
            if not torch.isfinite(loss): raise ValueError('Non-finite TARGO training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise ValueError('Non-finite TARGO gradient')
            optimizer.step(); steps+=1
            result['losses'].append(dict(epoch=epoch+1,step=steps,total=loss.item(),
                examples=len(y[0]),positive_labels=int(y[0].sum().item()),**{k:v.item() for k,v in parts.items()}))
            print(json.dumps(result['losses'][-1]),flush=True); write_result(out,result)
            if short and steps>=config.training_steps: break
        if val_data:
            rng = capture_rng_state(); np.random.seed(config.seed+100000+epoch); torch.manual_seed(config.seed+100000+epoch)
            net.eval(); totals={}; count=0; correct=0
            with torch.inference_mode():
                for batch,value in enumerate(torch.utils.data.DataLoader(val_data,batch_size=config.batch_size,num_workers=0)):
                    if config.eval_batch_limit and batch>=config.eval_batch_limit: break
                    x,y,pos = native.prepare_batch(value,torch.device('cuda'))
                    prediction = native.select(net(x,pos)); _,parts = native.loss_fn(prediction,y)
                    n = len(y[0]); count+=n; correct+=int((torch.round(prediction[0])==y[0]).sum().item())
                    for key,value in parts.items(): totals[key]=totals.get(key,0.)+value.item()*n
            if not count or not all(np.isfinite(v) for v in totals.values()): raise ValueError('Invalid TARGO validation loss')
            result['validation'].append(dict(epoch=epoch+1,examples=count,label_accuracy=correct/count,
                                              **{k:v/count for k,v in totals.items()}))
            restore_rng_state(rng)
        if (epoch+1)%10==0: scheduler.step()
        state = dict(format='grasppanda-targo-v1',model=net.state_dict(),optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),rng=capture_rng_state(),signature=expected,
                     completed_epochs=epoch+1,completed_steps=steps,resume_supported=not short)
        temporary = out/'checkpoint.pt.tmp'; torch.save(state,temporary); temporary.replace(out/'checkpoint.pt')
        result.update(checkpoint='checkpoint.pt',completed_epochs=epoch+1,completed_steps=steps)
        write_result(out,result)
        if short and steps>=config.training_steps: break
    return result


def run(config,out):
    repo = prepare()
    return infer(config,out,repo) if config.action=='infer' else train(config,out,repo,short=config.action=='train_short')
