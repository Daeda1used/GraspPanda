"""Native SpaHybGen voxel target preparation and contact/wrench training."""
import ast
import importlib.util
import sys
import time


def run(config,out,steps=3):
    import numpy as np
    import torch
    from grasppanda.worker import prepare
    from grasppanda.jobs import digest
    repo=prepare('spahybgen');sys.path.insert(0,str(repo/'src'))
    prepared=out/'prepared';prepared.mkdir()
    script=repo/'scripts/generate_dataset.py'
    tree=ast.parse(script.read_text())
    replacements={'start_scene_index':config.scene,'end_scene_index':config.scene+1,
                  'annIds_task':[config.frame,config.frame+1,1],'camera':config.camera}
    class Settings(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
                node.value=ast.parse(repr(replacements[node.targets[0].id]),mode='eval').body
            return node
    tree=Settings().visit(tree)
    sys.argv=[str(script),'--graspnet',config.dataset_root,'--output',str(prepared)]
    exec(compile(ast.fix_missing_locations(tree),str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
    from spahybgen.dataset import Dataset
    from spahybgen.networks import load_network
    spec=importlib.util.spec_from_file_location('grasppanda_native_spahyb_trainer',repo/'scripts/train_shgn.py')
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    dataset=Dataset(prepared,numsample=2000,orientation_type='quat',grid_type='voxel',data_type='Indexed')
    batch=Dataset.collate_fn_concatenate([dataset[0]])
    x,y,index=training.prepare_batch_concatenate(batch,torch.device('cuda'),'Indexed')
    if not (y[0]>0).any() or not (y[2]>0).any():raise ValueError('Prepared contact or wrench labels contain no positive targets')
    model=load_network(config.checkpoint,torch.device('cuda'),dict(voxel_discreteness=80,orientation='quat',augment=False)).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    losses=[];updates=[];start=time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred=training.select_concatenate(model(x),index)
        total,*metrics=training.loss_fn(pred,y,'CEL','quat','CEL')
        parts=dict(contact_bce=float(training._qual_loss_fn(pred[0],y[0],'CEL').mean().detach()),
            weighted_rotation=float((y[0]*training._rot_loss_fn(pred[1],y[1],'quat')[0]).mean().detach()),
            wrench_bce=float(training._wrench_loss_fn(pred[2],y[2],'CEL').mean().detach()))
        if not torch.isfinite(total) or not all(np.isfinite(v) for v in parts.values()):raise ValueError('Non-finite native SpaHybGen loss')
        total.backward()
        params=[p for p in model.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Invalid SpaHybGen gradients')
        selected=[p for p in params if torch.count_nonzero(p.grad)]
        if not selected:raise ValueError('No nonzero SpaHybGen gradients')
        parameter=selected[0];before=parameter.detach().clone()
        norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        optimizer.step();delta=float((parameter.detach()-before).norm())
        if not delta>0 or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Invalid SpaHybGen parameter update')
        losses.append(dict(total=float(total.detach()),components=parts));updates.append(dict(gradient_norm=norm,parameter_update_norm=delta))
        print('LOSS',losses[-1],flush=True)
    torch.save(model.state_dict(),out/'checkpoint.pt');torch.save(optimizer.state_dict(),out/'optimizer.pt')
    return dict(method='spahybgen',stage='real_voxel_contact_wrench_training',camera=config.camera,scene=config.scene,frame=config.frame,
        optimizer_steps=steps,losses=losses,updates=updates,seconds=time.monotonic()-start,
        positive_contact_samples=int((y[0]>0).sum()),positive_wrench_samples=int((y[2]>0).sum()),
        prepared_label_sha256={str(p.relative_to(prepared)):digest(p) for p in prepared.rglob('*') if p.is_file()},
        checkpoint_sha256=digest(config.checkpoint),
        protocol='Original single-frame generator: 0.4 m / 80-cell TSDF and voxel volume, native table shift, friction 0.2 grasps sampled at 1/8, native tip/wrench conversion. Released voxel UNet, indexed 2000-point sampling, native contact BCE + score-weighted quaternion + wrench BCE; batch 1, fixed real sample, Adam. No robot-hand optimization or full training convergence.',ap=None)
