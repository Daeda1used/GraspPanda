"""HGGD epoch training with native geometry, explicit stages and strict resume."""
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import random
import sys


class Frames:
    def __init__(self, dataset, indices, camera, augmentation):
        self.dataset, self.indices, self.camera, self.augmentation = dataset, tuple(indices), camera, augmentation
    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.dataset[self.indices[index]]
    def __getattr__(self, name): return getattr(object.__getattribute__(self, 'dataset'), name)
    def setaug(self):
        custom = self.augmentation and self.augmentation.get('mode', 'custom') != 'native'
        return self.dataset.unaug() if custom else self.dataset.setaug()


def worker_init(_worker):
    import numpy as np
    import torch
    from functools import partial
    camera = torch.utils.data.get_worker_info().dataset.camera
    configuration = importlib.import_module('dataset.config')
    intrinsic = partial(configuration.get_camera_intrinsic, camera=camera)
    for name, module in list(sys.modules.items()):
        if name.startswith('dataset.') and module is not None:
            if hasattr(module, 'camera'): module.camera = camera
            if hasattr(module, 'get_camera_intrinsic'): module.get_camera_intrinsic = intrinsic
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)


def inventory(*datasets):
    """Detect changed frame inventories without reading all image/label bytes."""
    digest = hashlib.sha256()
    for data in datasets:
        for index in data.indices:
            for field in ('colorpath', 'depthpath', 'grasppath'):
                path = Path(getattr(data.dataset, field)[index])
                if not path.is_file(): raise ValueError('Required HGGD frame missing: ' + str(path))
                stat = path.stat()
                digest.update(json.dumps((str(path), stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


class Accumulation:
    def __init__(self, models, optimizer, scheduler, count):
        self.models, self.optimizer, self.scheduler, self.count = models, optimizer, scheduler, count
        self.pending = self.batches = self.completed = 0
        self.losses, self.updates = [], []

    def backward(self, loss, epoch):
        import torch
        if not torch.isfinite(loss): raise ValueError('Non-finite HGGD training loss')
        self.losses.append(dict(epoch=epoch, total=float(loss.detach())))
        loss.backward()
        self.pending += 1; self.batches += 1
        if self.pending == self.count: self.flush()

    def flush(self):
        if not self.pending: return
        import torch
        named = [(group+'.'+name, p) for group, model in self.models.items()
                 for name, p in model.named_parameters() if p.grad is not None]
        if not named or not all(torch.isfinite(p.grad).all() for _, p in named):
            raise ValueError('HGGD gradients are missing or non-finite')
        for _, p in named: p.grad.div_(self.pending)
        norm = float(torch.sqrt(sum(p.grad.square().sum() for _, p in named)))
        torch.nn.utils.clip_grad_value_([p for _, p in named], 1.)
        snapshots = {}
        for name, p in named:
            branch = name.split('.')[0]
            if branch not in snapshots and p.grad.count_nonzero(): snapshots[branch] = (p, p.detach().clone())
        if not snapshots: raise ValueError('HGGD gradients are all zero')
        rate = float(self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
        updates = {name: float((p.detach()-before).norm()) for name, (p, before) in snapshots.items()}
        if not any(v > 0 for v in updates.values()) or not all(torch.isfinite(p).all() for model in self.models.values() for p in model.parameters()):
            raise ValueError('HGGD parameters did not receive a finite update')
        self.updates.append(dict(learning_rate=rate, accumulated_batches=self.pending,
                                 gradient_norm=norm, parameter_update_norms=updates))
        print('TRAINING_UPDATE', json.dumps(self.updates[-1]), flush=True)
        self.pending = 0; self.completed += 1
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None: self.scheduler.step()


def run(config, out):
    from functools import partial
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .worker import prepare
    from .components import configure_model, load_checkpoint
    from .hggd_options import resolved
    from .hggd_driver import compile_functions
    from .image_augmentation import configure_dataset
    from .image_losses import ImageLosses
    from .optimization import build_optimizer, UpdateSchedule
    from .native_training import verify_restored_state
    from .jobs import digest
    options = resolved(config)
    out = Path(out); prepare('hggd')
    camera = importlib.import_module('dataset.config')
    camera.camera = config.camera
    camera.get_camera_intrinsic = partial(camera.get_camera_intrinsic, camera=config.camera)
    sys.argv = ['train_graspnet.py', '--optim', 'adamw']
    native = importlib.import_module('train_graspnet')
    args = native.parse_args()
    args.batch_size=config.batch_size; args.all_points_num=config.num_points; args.lr=config.learning_rate
    args.joint_trainning=options['joint_training']; args.pre_epochs=options['pre_epochs']; args.shift_epoch=options['shift_epochs']
    args.center_num=options['center_num']; args.group_num=options['group_num']; args.local_grasp_num=options['local_grasp_num']
    args.loc_a=1.; args.reg_b=5.; args.cls_c=1.; args.offset_d=1.
    labelroot = Path(config.label_root or Path(config.dataset_root)/'HGGD_Preprocessed'/f'6dto2drefine_{config.camera}')
    def dataset(scenes, training):
        data = native.GraspnetPointDataset(config.num_points, str(labelroot), config.dataset_root, scenes,
            noise=0, sigma=args.sigma, ratio=args.ratio, anchor_k=args.anchor_k, anchor_z=args.anchor_z,
            anchor_w=args.anchor_w, grasp_count=args.grasp_count, output_size=(640,360), random_rotate=False, random_zoom=False)
        data.is_aug=False
        if training: configure_dataset(data, config)
        else: data.eval()
        start = config.scene*256+config.frame if training and config.train_batch_limit else 0
        limit = config.train_batch_limit*config.batch_size if training else config.eval_batch_limit
        indices = range(start, min(len(data), start+limit)) if limit else range(len(data))
        return Frames(data, indices, config.camera, config.augmentation if training else {})
    train_data, val_data = dataset(list(range(100)), True), dataset([100], False)
    manifest = inventory(train_data, val_data)
    generator = torch.Generator()
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True,
        num_workers=config.data_workers, pin_memory=True, generator=generator, worker_init_fn=worker_init,
        persistent_workers=False, **({'multiprocessing_context':'spawn'} if config.data_workers else {}))
    validation = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    if not len(loader) or not len(validation): raise ValueError('HGGD training/validation range is empty')
    per_epoch = math.ceil(len(loader)/options['accumulation_steps'])
    anchor = native.AnchorGraspNet(in_dim=4, ratio=8, anchor_k=6)
    changed = configure_model(anchor, 'hggd', config.modules)
    local = native.PointMultiGraspNet(3, 49)
    anchor.cuda(); local.cuda()
    basic = torch.linspace(-1,1,8,device='cuda'); basic=(basic[1:]+basic[:-1])/2
    anchors = {'gamma':basic.clone(), 'beta':basic.clone()}
    payload = torch.load(config.checkpoint, map_location='cpu', weights_only=True) if config.checkpoint else None
    if payload is not None and (not isinstance(payload, dict) or not {'anchor','local','gamma','beta'} <= payload.keys()):
        raise ValueError('HGGD checkpoints require anchor/local networks and gamma/beta anchors')
    if payload:
        transfer = load_checkpoint(anchor, payload['anchor'], changed, config.checkpoint_policy)
        local.load_state_dict({k:v for k,v in payload['local'].items() if k.rsplit('.',1)[-1] not in ('total_ops','total_params')}, strict=True)
        anchors = {k:payload[k].cuda() for k in ('gamma','beta')}
    else:
        from .modules.dino import DinoPyramid
        pretrained = {}
        for prefix in changed:
            module = anchor.get_submodule(prefix.rstrip('.'))
            if isinstance(module, DinoPyramid):
                record=module.initialize_pretrained()
                if record: pretrained[prefix.rstrip('.')]=record
        transfer = dict(policy='constructor', initialized=changed, pretrained=pretrained)
    if any(v.shape != (7,) or not torch.isfinite(v).all() for v in anchors.values()):
        raise ValueError('HGGD requires finite seven-bin gamma/beta anchors')
    parameters = itertools.chain(anchor.parameters(), local.parameters())
    optimizer = build_optimizer(parameters, config) if config.optimizer else native.get_optimizer(args, parameters)
    scheduler = UpdateSchedule(optimizer, config.scheduler, per_epoch*config.epochs) if config.scheduler else torch.optim.lr_scheduler.StepLR(optimizer,5,.1)
    bnm = native.BNMomentumScheduler(anchor, bn_lambda=lambda epoch:max(.5*.5**(int(epoch/2)),.001),last_epoch=-1)
    objectives=ImageLosses(importlib.import_module('models.losses'),config)
    native.compute_anchor_loss=objectives.anchor; native.compute_multicls_loss=objectives.local
    state = Accumulation({'anchor':anchor,'local':local}, optimizer, scheduler if config.scheduler else None, options['accumulation_steps'])
    resumed = config.train_checkpoint_mode == 'resume'; first_epoch=0
    if resumed:
        if not payload or payload.get('format') != 'grasppanda_hggd_epoch_v1':
            raise ValueError('HGGD resume requires a complete GraspPanda epoch checkpoint; use initialize for author weights')
        required={'config','resolved_trainer','data_inventory','epoch','completed_updates','optimizer_state_dict','scheduler_state_dict','bn_scheduler_epoch'}
        if not required <= payload.keys(): raise ValueError('HGGD resume checkpoint is missing training state: '+str(sorted(required-payload.keys())))
        previous=payload['config']
        for key in ('dataset','dataset_root','label_root','method','modules','camera','num_points','batch_size','train_batch_limit','eval_batch_limit',
                    'scene','frame','seed','data_workers','loss','augmentation','optimizer','scheduler','learning_rate','trainer'):
            if previous.get(key) != config.to_dict()[key]: raise ValueError('HGGD resume configuration differs at '+key)
        if config.scheduler and previous['epochs'] != config.epochs: raise ValueError('Configured scheduling requires the original final-epoch horizon')
        if payload['resolved_trainer'] != options or payload['data_inventory'] != manifest: raise ValueError('HGGD resume trainer defaults or frame inventory changed')
        first_epoch=payload['epoch']; state.completed=payload['completed_updates']
        if type(first_epoch) is not int or type(state.completed) is not int or not 1 <= first_epoch < config.epochs or state.completed != first_epoch*per_epoch:
            raise ValueError('HGGD resume epoch/update counters are inconsistent or training is already complete')
        optimizer.load_state_dict(payload['optimizer_state_dict']); scheduler.load_state_dict(payload['scheduler_state_dict'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if (config.scheduler and scheduler.completed != state.completed) or (not config.scheduler and scheduler.last_epoch != first_epoch):
            raise ValueError('HGGD scheduler and optimizer update counters are inconsistent')
        if payload['bn_scheduler_epoch'] != first_epoch-1: raise ValueError('HGGD BN scheduler epoch is inconsistent')
        bnm.step(payload['bn_scheduler_epoch'])
        verify_restored_state(anchor.state_dict(),payload['anchor'],'anchor')
        verify_restored_state(local.state_dict(),payload['local'],'local')
        verify_restored_state(optimizer.state_dict(),payload['optimizer_state_dict'],'optimizer')
        verify_restored_state(scheduler.state_dict(),payload['scheduler_state_dict'],'scheduler')
        verify_restored_state(anchors,{k:payload[k] for k in anchors},'anchors')
    shifts=[]
    def shift(labels, current):
        before={k:v.clone() for k,v in current.items()}
        result=native.shift_anchors(labels,current)
        if not all(torch.isfinite(v).all() for v in result.values()): raise ValueError('Native anchor shift produced non-finite anchors')
        shifts.append(dict(labels=len(labels),delta={k:float((v-before[k]).norm()) for k,v in result.items()}))
        return result
    train, validate = compile_functions(native, state.backward, shift, options['shift_min_labels'])
    def save(path, epoch):
        path.parent.mkdir(parents=True,exist_ok=True)
        checkpoint=dict(format='grasppanda_hggd_epoch_v1', anchor=anchor.state_dict(),local=local.state_dict(),**anchors,
            optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),bn_scheduler_epoch=bnm.last_epoch,
            epoch=epoch,completed_updates=state.completed,config=config.to_dict(),resolved_trainer=options,data_inventory=manifest)
        temporary=path.with_suffix(path.suffix+'.tmp');torch.save(checkpoint,temporary);temporary.replace(path)
    def serial(value):
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        if hasattr(value,'tolist'):return value.tolist()
        return value
    epochs=[]
    for epoch in range(first_epoch,config.epochs):
        random.seed(config.seed+epoch);np.random.seed(config.seed+epoch);torch.manual_seed(config.seed+epoch);generator.manual_seed(config.seed+epoch)
        start_batch=state.batches;start_update=state.completed;start_loss=len(state.losses)
        training=train(epoch,anchor,local,loader,optimizer,anchors,args)
        state.flush()
        if state.batches-start_batch != len(loader) or state.completed-start_update != per_epoch:
            raise ValueError('HGGD epoch did not optimize every scheduled batch')
        training['loss']=sum(r['total'] for r in state.losses[start_loss:])/len(loader)
        if not config.scheduler:scheduler.step()
        bnm.step()
        # Persist the completed training epoch before potentially lengthy validation.
        save(out/'training/checkpoints'/f'epoch_{epoch+1:04d}.tar',epoch+1)
        save(out/'checkpoint.pt',epoch+1)
        random.seed(config.seed+100000+epoch);np.random.seed(config.seed+100000+epoch);torch.manual_seed(config.seed+100000+epoch)
        metrics=validate(epoch,anchor,local,validation,anchors,args)
        row=dict(epoch=epoch+1,training=serial(training),validation=serial(metrics));epochs.append(row)
        print('EPOCH',json.dumps(row),flush=True)
    return dict(stage='epoch_training',method='hggd',epochs=epochs,losses=state.losses,updates=state.updates,
        completed_updates=state.completed,resume_state_verified=resumed,anchor_shifts=shifts,modules=config.modules,
        checkpoint_transfer=transfer,resolved_trainer=options,loss_config=config.loss,augmentation=config.augmentation,
        data_inventory=manifest,checkpoint='checkpoint.pt',checkpoint_sha256=digest(out/'checkpoint.pt'),ap=None,
        protocol_note='Native HGGD training geometry/targets and scene-100 validation metrics; mean gradient accumulation flushes every configured group and the final partial group. Training uses full batches with deterministic epoch seeds. Validation metrics are native geometric matching, not official GraspNet AP.')
