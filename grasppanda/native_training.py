"""Native epoch trainers with composed models and bounded, lazy label loading."""
from collections import OrderedDict
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys
import time


def verify_restored_state(actual, expected, path='checkpoint'):
    """Require exact restored model/optimizer values before the next update."""
    import torch
    if isinstance(expected, torch.Tensor):
        equal = (isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                 and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    elif isinstance(expected, Mapping):
        equal = isinstance(actual, Mapping) and actual.keys() == expected.keys()
        if equal:
            for key in expected:verify_restored_state(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        equal = isinstance(actual, type(expected)) and len(actual) == len(expected)
        if equal:
            for index, (a, b) in enumerate(zip(actual, expected)):verify_restored_state(a, b, f'{path}[{index}]')
    else:
        equal = actual == expected
    if not equal:raise ValueError(f'Restored training state differs at {path}')


class LabelCache(Mapping):
    """Read native arrays on demand; cap retained objects/scenes per worker."""
    def __init__(self, root, kind, keys, capacity=2):
        self.root=Path(root);self.kind=kind;self._keys=tuple(keys)
        self.capacity=capacity;self.cache=OrderedDict()

    def __iter__(self):return iter(self._keys)
    def __len__(self):return len(self._keys)
    def __contains__(self,key):return key in self._keys

    def __getitem__(self,key):
        import numpy as np
        if key not in self:raise KeyError(key)
        if key not in self.cache:
            if self.kind=='collision':
                with np.load(self.root/'collision_label'/key/'collision_labels.npz') as data:
                    value={i:data[f'arr_{i}'] for i in range(len(data))}
            else:
                sparse=self.kind=='simplified'
                name=f'{key-1:03d}'
                with np.load(self.root/('grasp_label_simplified' if sparse else 'grasp_label')/(name+'_labels.npz')) as data:
                    value=tuple(data[k].astype(np.float32) for k in (('points','width','scores') if sparse else ('points','offsets','scores')))
                if not sparse:value+=(np.load(self.root/'tolerance'/(name+'_tolerance.npy')),)
            self.cache[key]=value
            while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def run(config,out):
    import numpy as np
    import torch
    from .worker import prepare
    from .components import configure_model,load_checkpoint
    from .jobs import digest
    from .training_options import configure_dataset,weighted_loss
    from .optimization import build_optimizer,UpdateSchedule,NativeScheduleDisabled,native_driver
    implementation_hash=digest(__file__)
    repo=prepare(config.method)
    economic=config.method=='economicgrasp'
    sparse=config.method=='graspness'
    args=[str(repo/'train.py'),'--dataset_root',config.dataset_root,'--camera',config.camera,
          '--log_dir',str(out/'training'),'--num_point',str(config.num_points),
          '--batch_size',str(config.batch_size),'--max_epoch',str(config.epochs),'--learning_rate',str(config.learning_rate)]
    if sparse:args+=['--voxel_size',str(config.voxel_size),'--model_name','grasppanda']
    if economic:args+=['--voxel_size',str(config.voxel_size),'--model','economicgrasp']
    if config.checkpoint and config.train_checkpoint_mode=='resume':
        args+=['--checkpoint_path',config.checkpoint]
        if sparse or economic:args+=['--resume']
    # EconomicGrasp parses arguments when its dataset/model modules import.
    sys.argv=args
    dataset_names=['dataset.graspnet_dataset'] if sparse or economic else ['graspnet_dataset']
    if config.method=='scale_balanced_grasp':dataset_names+=['graspnet_wonoise_dataset']
    keys=list(range(1,89)) if sparse else [i for i in range(1,89) if i!=19]
    labels=LabelCache(config.dataset_root,'simplified' if sparse else 'full',keys)
    def load_labels(root):return labels if sparse else (keys,labels)
    for name in dataset_names:
        module=importlib.import_module(name)
        if not economic:module.load_grasp_labels=load_labels
        native=getattr(module,'GraspNetDataset',None)
        if native is None:continue
        def lazy_dataset(*args,_native=native,**kwargs):
            if economic:
                # Native EconomicGrasp stores paths and opens scene labels per item.
                return configure_dataset(_native(*args,**kwargs),config)
            enabled=kwargs.pop('load_label',True)
            dataset=_native(*args,load_label=False,**kwargs)
            dataset.load_label=enabled
            if enabled:dataset.collision_labels=LabelCache(config.dataset_root,'collision',dataset.sceneIds)
            if dataset.split=='train':dataset=configure_dataset(dataset,config)
            return dataset
        module.GraspNetDataset=lazy_dataset

    model_module=importlib.import_module('models.economicgrasp' if economic else ('models.graspnet' if sparse else 'graspnet'))
    class_name='economicgrasp' if economic else ('GraspNet_MSCQ' if config.method=='scale_balanced_grasp' else 'GraspNet')
    native_class=getattr(model_module,class_name)
    changed=[];transfer=None
    def composed(*args,**kwargs):
        nonlocal changed,transfer
        model=native_class(*args,**kwargs)
        changed=configure_model(model,config.method,config.modules,config.voxel_size)
        if config.checkpoint and config.train_checkpoint_mode=='initialize':
            state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
            transfer=load_checkpoint(model,state.get('model_state_dict',state),changed,config.checkpoint_policy)
        return model
    setattr(model_module,class_name,composed)

    resume=None
    if config.checkpoint and config.train_checkpoint_mode=='resume':
        resume=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
        for key in ('model_state_dict','optimizer_state_dict','epoch'):
            if key not in resume:raise ValueError(f'Resume checkpoint missing {key}; use initialize to load model weights only')
        if int(resume['epoch'])>=config.epochs:raise ValueError('Final epoch must exceed the resume checkpoint epoch')
        previous=resume.get('config')
        if previous:
            for key in ('dataset','method','modules','camera','num_points','voxel_size','batch_size','train_batch_limit','scene','frame','seed','data_workers','loss','augmentation','optimizer','scheduler','learning_rate'):
                if previous.get(key,{} if key in ('loss','augmentation','optimizer','scheduler') else None)!=config.to_dict()[key]:raise ValueError(f'Resume configuration differs at {key}; use initialize for a new experiment')
            if config.method=='scale_balanced_grasp' and previous.get('epochs')!=config.epochs:
                raise ValueError('Scale-Balanced-Grasp OneCycle resume requires its original final-epoch horizon')
            if economic and not config.scheduler and previous.get('epochs')!=config.epochs:
                raise ValueError('EconomicGrasp native cosine resume requires its original final-epoch horizon')
            if config.scheduler and previous.get('epochs')!=config.epochs:
                raise ValueError('Configured scheduling requires the original final-epoch horizon when resuming')
        elif config.optimizer or config.scheduler or config.loss or config.augmentation:
            raise ValueError('Resume with training overrides requires saved configuration metadata')
        if config.scheduler and 'scheduler_state_dict' not in resume:
            raise ValueError('Resume checkpoint is missing the configured scheduler state')

    # Use the native sampler and collator. Explicit batch limits select a
    # prefix of real frames, while preserving augmentation and epoch logic.
    native_loader=torch.utils.data.DataLoader
    def loader(dataset,*args,**kwargs):
        train=dataset.split=='train'
        limit=config.train_batch_limit if train else config.eval_batch_limit
        if limit:
            start=config.scene*256+config.frame if train else 0
            stop=min(start+limit*config.batch_size,len(dataset))
            if stop<=start:raise ValueError('Bounded training frame range is empty')
            dataset=torch.utils.data.Subset(dataset,range(start,stop))
        kwargs['num_workers']=config.data_workers
        from .pcm_options import selected as pcm_selected
        from .ptv2_options import selected as ptv2_selected
        if train and (pcm_selected(config) or ptv2_selected(config)):
            if len(dataset) < config.batch_size:
                raise ValueError('The selected point encoder requires at least one full training batch')
            # The native coarse/global BatchNorm cannot train on a singleton
            # remainder; keep batch composition stable across epoch resumes.
            kwargs['drop_last']=True
        return native_loader(dataset,*args,**kwargs)
    namespace={'__name__':'grasppanda_native_trainer','__file__':str(repo/'train.py')}
    namespace.update(_grasppanda_optimizer=lambda parameters:build_optimizer(parameters,config),
                     _grasppanda_disabled_schedule=NativeScheduleDisabled,
                     _grasppanda_loader=loader)
    source=(repo/'train.py').read_text()
    program=native_driver(source,repo/'train.py',config,adapt_loader=True)
    exec(program,namespace)
    model=namespace['net'];optimizer=namespace['optimizer']
    if config.scheduler:
        scheduler=UpdateSchedule(optimizer,config.scheduler,len(namespace['TRAIN_DATALOADER'])*config.epochs)
        namespace['adjust_learning_rate']=lambda *args:None
        namespace['get_current_lr']=lambda *args:optimizer.param_groups[0]['lr']
    else:scheduler=namespace.get('scheduler')
    if resume is not None and scheduler is not None and 'scheduler_state_dict' in resume:
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        # The native scheduler constructor changes optimizer LR/momentum.
        # Restore the exact saved values after rebuilding its schedule.
        optimizer.load_state_dict(resume['optimizer_state_dict'])
    if resume is not None:
        verify_restored_state(model.state_dict(), resume['model_state_dict'], 'model')
        verify_restored_state(optimizer.state_dict(), resume['optimizer_state_dict'], 'optimizer')
        if scheduler is not None and 'scheduler_state_dict' in resume:
            verify_restored_state(scheduler.state_dict(), resume['scheduler_state_dict'], 'scheduler')
    native_epoch=namespace['train_one_epoch']
    def seeded_epoch():
        # Epoch-local seeds also make an epoch-boundary resume repeatable.
        seed=config.seed+namespace['EPOCH_CNT']
        np.random.seed(seed % 2**32);torch.manual_seed(seed)
        return native_epoch()
    namespace['train_one_epoch']=seeded_epoch
    losses=[];updates=[];evaluations=[]
    loss_name='get_loss_economicgrasp' if economic else 'get_loss'
    original_loss=namespace[loss_name];original_step=optimizer.step
    def measured_loss(*args,**kwargs):
        loss,end=original_loss(*args,**kwargs)
        loss,end=weighted_loss(loss,end,config)
        parts={k:float(v.detach()) for k,v in end.items() if 'loss' in k.lower() and isinstance(v,torch.Tensor) and v.numel()==1}
        if not torch.isfinite(loss) or not all(np.isfinite(v) for v in parts.values()):raise ValueError('Non-finite native epoch loss')
        undefined=[k for k,v in end.items() if isinstance(v,torch.Tensor) and v.numel()==1 and ('acc' in k or 'prec' in k or 'recall' in k) and not torch.isfinite(v)]
        record=dict(epoch=namespace['EPOCH_CNT'],total=float(loss.detach()),components=parts,undefined_metrics=undefined)
        (losses if model.training else evaluations).append(record)
        print('LOSS',json.dumps(record),flush=True)
        return loss,end
    def measured_step(*args,**kwargs):
        used_lr=optimizer.param_groups[0]['lr']
        params=[p for p in model.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Invalid native epoch gradients')
        selected=[p for p in params if torch.count_nonzero(p.grad)]
        if not selected:raise ValueError('All native epoch gradients are zero')
        snapshots={'model':(selected[0],selected[0].detach().clone())}
        for prefix in changed:
            subset=[p for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None and torch.count_nonzero(p.grad)]
            if not subset:raise ValueError(f'No gradient reaches {prefix}')
            snapshots[prefix]=(subset[0],subset[0].detach().clone())
        norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        result=original_step(*args,**kwargs)
        delta={name:float((p.detach()-before).norm()) for name,(p,before) in snapshots.items()}
        if not all(v>0 for v in delta.values()) or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Invalid native epoch parameter update')
        updates.append(dict(gradient_norm=norm,parameter_update_norms=delta,learning_rate=used_lr))
        if config.scheduler:scheduler.step()
        return result
    namespace[loss_name]=measured_loss;optimizer.step=measured_step
    # The author checkpoints omit composition metadata. Add it to each native
    # epoch checkpoint so resumed runs retain an auditable contract.
    original_save=torch.save
    def save(value,path,*args,**kwargs):
        if isinstance(value,dict) and 'model_state_dict' in value:
            value=dict(value,config=config.to_dict())
            if scheduler is not None:value['scheduler_state_dict']=scheduler.state_dict()
            archive=out/'training/checkpoints'/f"epoch_{int(value['epoch']):04d}.tar"
            archive.parent.mkdir(parents=True,exist_ok=True)
            original_save(value,archive,*args,**kwargs)
        return original_save(value,path,*args,**kwargs)
    torch.save=save
    start=time.monotonic()
    try:namespace['train'](namespace['start_epoch'])
    finally:
        torch.save=original_save
        for name in ('TRAIN_WRITER','TEST_WRITER','LOG_FOUT'):
            if name in namespace:namespace[name].close()
    torch.cuda.synchronize()
    paths=list((out/'training/checkpoints').glob('epoch_*.tar'))
    if not updates or not paths:raise ValueError('Native trainer did not produce updates and a checkpoint')
    import shutil
    shutil.copyfile(max(paths,key=lambda path:int(path.stem.removeprefix('epoch_'))),out/'checkpoint.pt')
    if transfer:(out/'component_transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
    return dict(stage='native_epoch_training',method=config.method,modules=config.modules,augmentation=config.augmentation,loss_config=config.loss,
        optimizer_config=config.optimizer,scheduler_config=config.scheduler,optimizer_class=type(optimizer).__name__,
        implementation_sha256=implementation_hash,
        checkpoint_mode=config.train_checkpoint_mode,resume_state_verified=resume is not None,
        start_epoch=namespace['start_epoch'],final_epoch=config.epochs,
        optimizer_steps=len(updates),losses=losses,updates=updates,evaluation_losses=evaluations,
        train_batch_limit=config.train_batch_limit,eval_batch_limit=config.eval_batch_limit,
        seconds=time.monotonic()-start,checkpoint_sha256=digest(out/'checkpoint.pt'),
        protocol='Native epoch driver with configured loss, augmentation and optimization settings; original optimizer/schedule when no override is supplied; lazy native labels. Nonzero batch limits restrict real-frame coverage and do not establish full-dataset convergence.',ap=None)
