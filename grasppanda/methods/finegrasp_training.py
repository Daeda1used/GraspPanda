"""FineGrasp training with native targets, objectives, optimization and model code."""
import json
from pathlib import Path


def seed_worker(_worker):
    import random
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def author_config():
    import importlib.util
    import sys
    from grasppanda.config import ROOT, catalogue
    path = ROOT/catalogue()['finegrasp']['path']/'projects/finegrasp_graspnet1b/configs/config_finegrasp_minkunet.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('_grasppanda_finegrasp_training_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config, out, steps=None):
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from grasppanda.methods.finegrasp import load_model
    from grasppanda.methods.finegrasp_data import FineGraspDataset
    from grasppanda.training.native import verify_restored_state
    from grasppanda.training.optimization import build_optimizer, UpdateSchedule
    from grasppanda.training.options import LOSS_TERMS, weighted_loss
    from grasppanda.jobs import digest
    short = steps is not None
    out = Path(out)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model, architecture, payload, changed, transfer = load_model(config, training=True)
    if architecture.get('num_view', 300) != 300 or architecture.get('num_angle', 12) != 12 or architecture.get('num_depth', 4) != 4:
        raise ValueError('FineGrasp training requires the released 300-view, 12-angle, 4-depth supervision')
    from robo_orchard_lab.dataset.collates import collate_batch_dict
    cache = Path(config.label_root) if config.label_root else out/'prepared/finegrasp'
    dataset = FineGraspDataset(config, cache, augment=not short or bool(config.augmentation))
    start = config.scene*256 + config.frame
    from grasppanda.components import requires_scene_batch
    multi_frame = requires_scene_batch(config)
    if short:
        dataset = Subset(dataset, range(start, start+config.batch_size) if multi_frame else [start]*config.batch_size)
    elif config.train_batch_limit:
        stop = min(start + config.train_batch_limit*config.batch_size, len(dataset))
        dataset = Subset(dataset, range(start, stop))
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=not short,
        num_workers=config.data_workers, collate_fn=collate_batch_dict, generator=generator,
        worker_init_fn=seed_worker, persistent_workers=False, drop_last=multi_frame,
        **({'multiprocessing_context': 'spawn'} if config.data_workers else {}))
    if not len(loader): raise ValueError('FineGrasp training range is empty')
    total = steps + config.proposal_warmup_steps if short else len(loader)*config.epochs
    model.cuda().train()
    native = author_config()
    trainer_cfg = SimpleNamespace(lr=config.learning_rate, max_step=total, max_epoch=1 if short else config.epochs)
    _, optimizer, native_scheduler = native.build_optimizer(model, trainer_cfg)
    schedule_function = native_scheduler.lr_lambdas[0]
    if config.optimizer: optimizer = build_optimizer(model.parameters(), config)
    # Keep the author's exact schedule while storing portable Python scalars.
    for group in optimizer.param_groups:
        group['lr'] = config.learning_rate
        group['initial_lr'] = config.learning_rate
    if config.scheduler:
        scheduler = UpdateSchedule(optimizer, config.scheduler, total)
    elif short:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: float(schedule_function(step)))
    resumed = config.train_checkpoint_mode == 'resume'
    first_epoch = 0
    completed = 0
    if resumed:
        if not payload or not payload.get('config') or 'optimizer_state_dict' not in payload:
            raise ValueError('FineGrasp resume requires a complete GraspPanda training checkpoint')
        previous = payload['config']
        for key in ('dataset','dataset_root','method','modules','camera','num_points','voxel_size','batch_size',
                    'train_batch_limit','scene','frame','seed','data_workers','loss','augmentation','optimizer','scheduler','learning_rate','epochs'):
            if previous.get(key) != config.to_dict()[key]:
                raise ValueError(f'FineGrasp resume configuration differs at {key}; use initialize for a new experiment')
        first_epoch = payload.get('epoch')
        completed = payload.get('completed_updates')
        if type(first_epoch) is not int or type(completed) is not int or first_epoch < 1 or completed != first_epoch*len(loader):
            raise ValueError('FineGrasp resume requires a complete epoch checkpoint with consistent update counters')
        if first_epoch >= config.epochs: raise ValueError('Final epoch must exceed the saved epoch')
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if scheduler is not None:
            if 'scheduler_state_dict' not in payload: raise ValueError('FineGrasp checkpoint is missing scheduler state')
            scheduler.load_state_dict(payload['scheduler_state_dict'])
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        verify_restored_state(model.state_dict(), payload['model_state_dict'], 'model')
        verify_restored_state(optimizer.state_dict(), payload['optimizer_state_dict'], 'optimizer')
        if scheduler is not None: verify_restored_state(scheduler.state_dict(), payload['scheduler_state_dict'], 'scheduler')
    from grasppanda.methods.seed_warmup import SeedWarmup, objective as seed_objective
    from robo_orchard_lab.models.finegrasp.losses import ObjectnessLoss, GraspnessLoss
    seed_losses = [function for function in model.loss if isinstance(function, (ObjectnessLoss, GraspnessLoss))]
    if len(seed_losses) != 2:
        raise ValueError('FineGrasp seed warmup requires its registered objectness and graspness losses')
    warming_up = False
    def valid_seeds(_module, _args, end):
        if warming_up:
            raise SeedWarmup(end)
        mask = (end['objectness_score'].argmax(1) == 1) & (end['graspness_score'].squeeze(1) > model.graspness_threshold)
        if not torch.all(mask.any(1)):
            raise ValueError('FineGrasp has no predicted graspable seeds in a batch item. Use a trained initialization or train the seed-prediction stage before the full grasp objective.')
    model.graspable.register_forward_hook(valid_seeds)
    def device(value):
        if isinstance(value, torch.Tensor): return value.cuda(non_blocking=True)
        if isinstance(value, dict): return {key: device(item) for key, item in value.items()}
        if isinstance(value, list): return [device(item) for item in value]
        if isinstance(value, tuple): return tuple(device(item) for item in value)
        return value
    terms = LOSS_TERMS['finegrasp']
    losses, updates = [], []
    def step(batch, epoch):
        nonlocal completed, warming_up
        warming_up = short and completed < config.proposal_warmup_steps
        optimizer.zero_grad(set_to_none=True)
        try:
            end = model(device(batch))
        except SeedWarmup as signal:
            end = signal.end_points
        if warming_up:
            for function in seed_losses:
                end = function(end)
            loss, end = seed_objective(end, config)
        else:
            native_loss = sum(coefficient*end[key] for key, coefficient in terms.values())
            loss, end = weighted_loss(native_loss, end, config)
        active_terms = ('objectness', 'graspness') if warming_up else terms
        parts = {name: float(end[terms[name][0]].detach()) for name in active_terms}
        if not torch.isfinite(loss) or not all(np.isfinite(value) for value in parts.values()):
            raise ValueError('FineGrasp produced a non-finite training loss')
        loss.backward()
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.grad is not None]
        if not named or not all(torch.isfinite(parameter.grad).all() for _, parameter in named):
            raise ValueError('FineGrasp gradients are missing or non-finite')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        from grasppanda.training.options import snapshot_components
        active_prefixes = [prefix for prefix in changed if prefix == 'backbone.'] if warming_up else changed
        _, zero_gradients = snapshot_components(model, active_prefixes)
        selected = [(name, parameter) for name, parameter in named if torch.count_nonzero(parameter.grad)]
        if not selected: raise ValueError('FineGrasp gradients are all zero')
        # Include zero-initialized biases: native warmup can start below the
        # representable update size of some nonzero float32 weights.
        snapshots = [(name, parameter, parameter.detach().clone()) for name, parameter in selected]
        rate = float(optimizer.param_groups[0]['lr'])
        optimizer.step()
        deltas = {name: float((parameter.detach()-before).norm()) for name, parameter, before in snapshots}
        if not any(value > 0 for value in deltas.values()) or not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError('FineGrasp parameters did not receive a finite update')
        component_updates = {prefix: sum(value for name, value in deltas.items() if name.startswith(prefix)) for prefix in active_prefixes}
        if not all(value > 0 for prefix, value in component_updates.items() if prefix not in zero_gradients):
            raise ValueError('A replaced FineGrasp component received no update')
        completed += 1
        if scheduler is not None: scheduler.step()
        row = dict(epoch=epoch, total=float(loss.detach()), components=parts)
        losses.append(row)
        updates.append(dict(gradient_norm=float(norm), learning_rate=rate, component_updates=component_updates, zero_gradient_components=zero_gradients,
                            parameter_update_norm=sum(deltas.values())))
        if config.proposal_warmup_steps:
            row['stage'] = updates[-1]['stage'] = 'Seed warmup' if warming_up else 'Grasp training'
        print('LOSS', json.dumps(row), flush=True)
    def save(path, epoch):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), epoch=epoch,
                     completed_updates=completed, config=config.to_dict(), model_config=architecture)
        if scheduler is not None: state['scheduler_state_dict'] = scheduler.state_dict()
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temporary)
        temporary.replace(path)
        (path.parent/'model.config.json').write_text(json.dumps(architecture, indent=2)+'\n')
    if short:
        generator.manual_seed(config.seed)
        for _ in range(total): step(next(iter(loader)), 0)
    else:
        for epoch in range(first_epoch, config.epochs):
            random.seed(config.seed+epoch)
            np.random.seed(config.seed+epoch)
            torch.manual_seed(config.seed+epoch)
            generator.manual_seed(config.seed+epoch)
            for batch in loader: step(batch, epoch)
            save(out/'training/checkpoints'/f'epoch_{epoch+1:04d}.tar', epoch+1)
    save(out/'checkpoint.pt', 0 if short else config.epochs)
    return dict(stage='labeled_training' if short else 'epoch_training', method='finegrasp',
        losses=losses, updates=updates, completed_updates=completed, resume_state_verified=resumed,
        proposal_warmup_steps=config.proposal_warmup_steps,
        checkpoint='checkpoint.pt', checkpoint_sha256=digest(out/'checkpoint.pt'),
        modules=config.modules, checkpoint_transfer=transfer, ap=None,
        protocol_note='Native FineGrasp economic labels and losses. Derived normals and instance-normalized graspness are cached locally; native flips transform normals with points. Full-split AP is evaluated separately.')
