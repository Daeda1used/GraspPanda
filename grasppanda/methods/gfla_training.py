"""GFLA native contact-map/SDF objective with isolated target preparation."""
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def _context():
    from grasppanda.config import ROOT
    from grasppanda.worker import prepare
    from grasppanda.overlays import prepare_overlay
    repo=prepare('gfla')
    for path in (prepare_overlay('gfla'),ROOT/'environments/extensions',prepare_overlay('gfla_source')):sys.path.insert(0,str(path))
    os.environ['GRASPPANDA_WEIGHTS_ROOT']=str(ROOT/'checkpoints/gfla')
    import grasp_nms_cpp
    sys.modules['models.my_grasp_nms.grasp_nms_cpp']=grasp_nms_cpp
    import models.contact_grasp as contact
    from grasppanda.sdf_queries import install_gfla_sampler
    install_gfla_sampler(contact)
    return repo,contact


def prepare_object(task):
    root,obj=task
    import torch
    torch.set_num_threads(1)
    _,contact=_context()
    api=contact.CttGraspNet(root,camera='realsense',split=[0])
    directory=str(Path(root)/api.ctt_grasp_group_dir)
    try:
        contact.ContactGraspGroupWithIndices.load(directory,f'{obj:03d}')
        return obj
    except (FileNotFoundError,ValueError,EOFError):
        contact.calcContactGraspgroup(api,obj,save_directory=directory)
        return obj


def run(config,out,steps=3):
    import numpy as np
    import torch
    import scipy.io
    from grasppanda.jobs import digest
    repo,contact=_context()
    from LauncherTemplate import load_yaml,LossManager
    import models.CFG_main as module
    root=Path(config.dataset_root);staging=out/'prepared';staging.mkdir(exist_ok=True)
    for name in ('scenes','models','grasp_label','collision_label'):
        (staging/name).symlink_to(root/name,target_is_directory=True)
    # Existing Dex-Net caches are read through individual links. New caches
    # remain inside this experiment if an object has no precomputed file.
    dex=staging/'dex_models';dex.mkdir()
    for path in (root/'dex_models').glob('*.pkl'):(dex/path.name).symlink_to(path)
    cache=staging/'ctt_grasp_label';cache.mkdir()
    if config.label_root:
        source=Path(config.label_root)
        for labels in source.glob('*_labels.npy'):
            indices=labels.with_name(labels.stem+'_indices.npy')
            if indices.is_file():
                a=np.load(labels,mmap_mode='r');b=np.load(indices,mmap_mode='r')
                if a.ndim!=2 or a.shape[1]!=11 or len(a)!=len(b):raise ValueError('Invalid reusable native GFLA contact cache')
                for path in (labels,indices):(cache/path.name).symlink_to(path.resolve())
    ids=(scipy.io.loadmat(root/'scenes'/f'scene_{config.scene:04d}'/config.camera/'meta'/f'{config.frame:04d}.mat')['cls_indexes'].reshape(-1)-1).astype(int).tolist()
    missing=[obj for obj in ids if not (cache/f'{obj:03d}_labels_indices.npy').is_file()]
    if missing:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        workers=min(config.data_workers or 4,len(missing))
        print(f'Preparing native contact labels for {len(missing)} objects with {workers} workers',flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            for obj in pool.map(prepare_object,[(str(staging),obj) for obj in missing]):
                print(f'Contact cache ready: object {obj:03d}',flush=True)
    api=contact.CttGraspNet(str(staging),camera=config.camera,split=[config.scene])
    contact_dataset=contact._ContactMapDataset([config.scene],api)
    index=config.frame*2+(config.camera=='realsense')
    sample=contact._ContactMapDataset_collate_fn([contact_dataset[index]])
    # Model-free predictor initialization in this LauncherTemplate revision
    # dereferences a None model. Its native compute/save methods need only dataset.
    predictor=SimpleNamespace(dataset=contact_dataset)
    with torch.no_grad():target=contact.calcContactMap_torch.run_model(predictor,sample)
    contact.calcContactMap_torch.save_results(predictor,[index],target)
    if not torch.isfinite(target).all():raise ValueError('Invalid generated native GFLA contact map')
    del target,sample,predictor
    base=module.GPCDataset_Grasp1B(str(staging),[config.scene],config.camera,check_completion=False)
    # Only the selected prefix of camera frames is prepared; the geometry and
    # visibility algorithms are unchanged. No writes enter source symlinks.
    base._get_annId_num=lambda scene:config.frame+1
    base._generate_down_sample_scene()
    dataset=module.GPCDataset(base)
    dataset.pre_load_scenes();dataset.set_return_input_2d(True)
    data=dataset[config.frame]
    batch=list(module.cfg_collate_fn([data]))
    # Native CFD expects a list of HWC NumPy images. Explicit conversion
    # avoids NumPy attempting scalar coercion of the loader's Torch images.
    batch[0]=[x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else x for x in batch[0]]
    cfg=load_yaml(str(repo/'cfg/config.yaml'))
    model=module.CFGNet(cfg).cuda().train()
    state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
    expected=set(model.state_dict());unused=sorted(set(state)-expected)
    model.load_state_dict({k:v for k,v in state.items() if k in expected},strict=True)
    model.csf_net.return_coord_only=True
    criterion=module.CFGLoss(cfg)
    parts={};native_loss=criterion.forward
    def measured(*args,**kwargs):
        values,watch=native_loss(*args,**kwargs)
        parts.clear();parts.update({f'native_loss_{i}':float(v.detach()) for i,v in enumerate(values)})
        if not all(np.isfinite(v) for v in parts.values()):raise ValueError('Non-finite GFLA loss component')
        return values,watch
    criterion.forward=measured
    launcher=SimpleNamespace(model=model,criterion=LossManager(criterion),config={'batch_size':1})
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    launcher.optimizer=optimizer
    losses=[];updates=[];start=time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss=module.CFG_Trainer.run_model(launcher,copy.deepcopy(batch))
        if not torch.isfinite(loss):raise ValueError('Non-finite GFLA objective')
        loss.backward()
        params=[p for p in model.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Invalid GFLA gradients')
        snapshots={}
        for prefix in ('cfd_net.','gpc_net.'):
            selected=[p for key,p in model.named_parameters() if key.startswith(prefix) and p.grad is not None and torch.count_nonzero(p.grad)]
            if not selected:raise ValueError('No gradient in '+prefix)
            snapshots[prefix]=(selected[0],selected[0].detach().clone())
        norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        # Same CSF clipping as CFG_Trainer.backward; contact/SDF objectives are
        # the released active losses, while token losses are commented upstream.
        torch.nn.utils.clip_grad_norm_(model.csf_net.parameters(),1e-10)
        optimizer.step()
        delta={key:float((p.detach()-old).norm()) for key,(p,old) in snapshots.items()}
        if not all(v>0 for v in delta.values()) or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Invalid GFLA parameter update')
        losses.append(dict(total=float(loss.detach()),components=dict(parts)))
        updates.append(dict(gradient_norm=norm,parameter_update_norms=delta))
        print('LOSS',losses[-1],flush=True)
    torch.save(model.state_dict(),out/'checkpoint.pt')
    torch.save(optimizer.state_dict(),out/'optimizer.pt')
    evidence={str(p.relative_to(staging)):digest(p) for folder in ('ctt_map_dir','downsampled_scenes') for p in (staging/folder).rglob('*.npz')}
    return dict(method='gfla',stage='real_label_contact_sdf_training',camera=config.camera,scene=config.scene,frame=config.frame,
        optimizer_steps=steps,losses=losses,updates=updates,seconds=time.monotonic()-start,prepared_label_sha256=evidence,
        checkpoint_sha256=digest(config.checkpoint),unused_checkpoint_keys=unused,
        protocol='Original real contact-grasp labels (batch 16, exact nearest-surface index, independent object preprocessing workers), contact-map, full-scene geometry and visibility target preparation; native CFG_Trainer.run_model, CFGLoss and LossManager. Batch 1, repeated real frame, no augmentation, four local volumes, reduced-rate Adam. Native CSF clipping retained; the released objective enables contact/SDF losses, not the commented token losses.',ap=None)
