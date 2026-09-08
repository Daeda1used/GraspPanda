"""Real-label checks for both CenterGrasp training stages."""
import json
from pathlib import Path
import sys
import time


def run(config,out,steps=3):
    import numpy as np
    import torch
    import mplib
    from .config import ROOT
    from .worker import prepare
    from .overlays import prepare_overlay
    from .jobs import digest
    prepare(config.method)
    # This author's fork has different sampling and watertight-mesh semantics.
    # Scope it to this worker rather than replacing the shared base package.
    sys.path.insert(0,str(ROOT/'environments/sources/centergrasp-mesh-to-sdf'))
    sys.modules['mplib.pymp.fcl']=mplib.pymp.collision_detection.fcl
    sys.path.insert(0,str(prepare_overlay('centergrasp')))
    from centergrasp.configs import Directories
    root=Path(config.dataset_root);staging=out/'prepared';staging.mkdir(exist_ok=True)
    Directories.GRASPNET=root
    Directories.SGDF_GRASPNET=staging/'sgdf'
    Directories.RGBD_GRASPNET=staging/'rgbd'
    import centergrasp.data_utils as du
    du.get_checkpoint_path=lambda name,folder:(Path(config.checkpoint) if folder=='ckpt_rgb' and name=='el6oa23g' else next((ROOT/'checkpoints/centergrasp'/folder/name).glob('*.ckpt')))
    from centergrasp.sgdf.training_deep_sgdf import load_sgdf_model
    from centergrasp.graspnet import sgdf_data
    from centergrasp.graspnet.sgdf_dataset import SGDFDatasetGraspnet
    # The native SGDF generator samples the real mesh and assigns the nearest
    # successful native grasp. A scene-wise SGDF folder is not this data format.
    for split in ('train','valid'):(Directories.SGDF_GRASPNET/split).mkdir(parents=True,exist_ok=True)
    sgdf_data.save_sgdf_data(0)
    sgdf,sgdf_specs=load_sgdf_model('6953cfxt')
    sgdf.train()
    dataset=SGDFDatasetGraspnet(config.num_points)
    sample=dataset[0]
    batch=[torch.as_tensor(sample[0],device='cuda').reshape(1)]+[v[None].cuda() for v in sample[1:]]
    all_losses=[];stage_results={};start=time.monotonic()

    def optimize(name,model,batch,component_names):
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        losses=[];updates=[]
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss=model.training_step(batch,step)
            if not torch.isfinite(loss):raise ValueError(f'{name}: non-finite native loss')
            loss.backward()
            params=[p for p in model.parameters() if p.grad is not None]
            if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError(f'{name}: invalid gradients')
            snapshots={}
            for component in component_names:
                selected=[p for key,p in model.named_parameters() if key.startswith(component) and p.grad is not None and torch.count_nonzero(p.grad)]
                if not selected:raise ValueError(f'{name}: no gradient in {component}')
                snapshots[component]=(selected[0],selected[0].detach().clone())
            norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
            optimizer.step()
            delta={key:float((p.detach()-old).norm()) for key,(p,old) in snapshots.items()}
            if not all(v>0 for v in delta.values()) or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError(f'{name}: invalid update')
            losses.append(dict(total=float(loss.detach()),stage=name,components=dict(component_losses)))
            updates.append(dict(gradient_norm=norm,parameter_update_norms=delta))
            print('LOSS',losses[-1],flush=True)
        import pytorch_lightning as pl
        path=out/('checkpoint.pt' if name=='rgb' else 'checkpoint_sgdf.pt')
        torch.save({'state_dict':model.state_dict(),'hyper_parameters':dict(model.hparams),
                    'pytorch-lightning_version':pl.__version__,'epoch':0,'global_step':steps,
                    'optimizer_states':[optimizer.state_dict()],'config':config.to_dict()},path)
        all_losses.extend(losses)
        return dict(optimizer_steps=steps,losses=losses,updates=updates,checkpoint_sha256=digest(path))

    component_losses={}
    native_sgdf_loss=sgdf.get_sgdf_loss
    def measure_sgdf(*args,**kwargs):
        values=native_sgdf_loss(*args,**kwargs)
        component_losses.clear();component_losses.update({k:float(v.detach()) if isinstance(v,torch.Tensor) else float(v) for k,v in zip(('grasp','sdf','code_regularization','total'),values)})
        return values
    sgdf.get_sgdf_loss=measure_sgdf
    stage_results['sgdf']=optimize('sgdf',sgdf,batch,('decoder.','embeddings.'))
    del sgdf,batch
    torch.cuda.empty_cache()

    from centergrasp.graspnet.make_heatmaps import make_heatmap
    make_heatmap(config.scene*256+config.frame)
    from centergrasp.graspnet.rgb_data import RGBDatasetGraspnet
    from centergrasp.rgb.training_centergrasp import load_rgb_model
    dataset=RGBDatasetGraspnet('6953cfxt',mode='train')
    batch=[v[None].cuda() for v in dataset[config.scene*256+config.frame]]
    rgb,_=load_rgb_model('el6oa23g');rgb.train()
    original_rgb_loss=rgb.compute_loss
    def measure_rgb(*args,**kwargs):
        values=original_rgb_loss(*args,**kwargs)
        component_losses.clear();component_losses.update({k:float(v.detach()) for k,v in zip(('total','heatmap','pose','shape'),values)})
        return values
    rgb.compute_loss=measure_rgb
    stage_results['rgb']=optimize('rgb',rgb,batch,('model.',))
    evidence={str(p.relative_to(staging)):digest(p) for p in staging.rglob('*') if p.is_file()}
    return dict(method='centergrasp',stage='real_label_two_stage_training',camera=config.camera,scene=config.scene,frame=config.frame,
        optimizer_steps=steps*2,steps_per_stage=steps,losses=all_losses,stages=stage_results,seconds=time.monotonic()-start,
        prepared_label_sha256=evidence,mesh_sha256=digest(root/'models/000/nontextured.ply'),
        grasp_label_sha256=digest(root/'grasp_label/000_labels.npz'),
        protocol='Two separate native training objectives: object 000 SGDF generated from its real mesh and successful grasps, then Kinect RGB heatmap/pose/shape training with native segmentation targets and released SGDF embeddings. Native training_step augmentation is retained. Batch 1, Adam with reduced learning rate; three steps per stage. Not joint training or convergence.',ap=None)
