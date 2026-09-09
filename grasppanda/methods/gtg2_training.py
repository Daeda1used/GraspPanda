"""Scene-disjoint graph ensembles with balanced sampling and epoch checkpoints."""
import bisect
import copy
from dataclasses import replace
import json
from pathlib import Path
import random

from grasppanda.methods.gtg2_options import resolved, TRAINER
from grasppanda.methods.gtg2_data import contract, cache_path, load_cache
from grasppanda.jobs import digest

FORMAT = 'grasppanda_gtg2_ensemble_v1'


def packet_paths(config, options):
    trainer = {**TRAINER, **config.trainer}
    result = []
    for scene in sorted(trainer['scenes']):
        seed = (config.seed + scene*256 + config.frame) % 2**32
        expected = contract(config.dataset_root, scene, config.camera, config.frame, options, seed)
        path = cache_path(config.label_root, expected)
        if not (path/'manifest.json').is_file():
            raise ValueError(f'GtG2 scene {scene:04d} is not prepared for this geometry/seed. Run ./panda prepare-gtg2 with the same experiment configuration.')
        load_cache(path, expected)
        result.append((scene, path))
    return result


class GraphDataset:
    def __init__(self, packets, options, augmentation, seed):
        import numpy as np
        self.packets, self.options, self.augmentation, self.seed = packets, options, augmentation, seed
        self.counts = [len(np.load(path/'targets.npy', mmap_mode='r')) for _, path in packets]
        self.ends = np.cumsum(self.counts).tolist()
        self.targets = np.concatenate([np.load(path/'targets.npy') for _, path in packets])
        self.scenes = np.concatenate([np.full(n, scene, dtype=np.int16) for (scene, _), n in zip(packets, self.counts)])
        self.epoch, self.fold, self._opened = 0, 0, {}

    def __len__(self): return int(self.ends[-1])

    def __getitem__(self, index):
        import numpy as np
        from grasppanda.modules.gtg2 import build_graph
        index = int(index)
        packet = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[packet-1] if packet else 0)
        if packet not in self._opened:
            if len(self._opened) >= 2: self._opened.pop(next(iter(self._opened)))
            path = self.packets[packet][1]
            self._opened[packet] = {n: np.load(path/(n+'.npy'), mmap_mode='r') for n in
                                   ('inside', 'outside', 'inside_offsets', 'outside_offsets')}
        data = self._opened[packet]
        regions = [data[n][data[n+'_offsets'][local]:data[n+'_offsets'][local+1]] for n in ('inside', 'outside')]
        seed = (self.seed + self.fold*1009 + self.epoch*104729 + index*17) % 2**32
        return build_graph(*regions, float(self.targets[index]), self.options, self.augmentation, seed)

    def __getstate__(self): return {**self.__dict__, '_opened': {}}


def training_indices(dataset, fold, epoch, trainer, seed):
    import numpy as np
    available = np.flatnonzero(dataset.scenes % 10 != fold)
    if trainer['sampling'] == 'all': return available
    rng = np.random.default_rng((seed + fold*1009 + (epoch//trainer['refresh_epochs'])*104729) % 2**32)
    scores = dataset.targets[available]
    selected = [available[scores >= 0]]
    for label in (-1., -.5):
        indices = available[scores == label]
        selected.append(rng.choice(indices, min(len(indices), trainer['negative_per_class']), replace=False))
    return np.concatenate(selected)


def batches(indices, size, limit=0, seed=None, training=False):
    import numpy as np
    indices = np.asarray(indices).copy()
    if seed is not None: np.random.default_rng(seed).shuffle(indices)
    if training and len(indices) < 2: raise ValueError('Each GtG2 training fold needs at least two prepared graphs')
    result = [indices[i:i+size].tolist() for i in range(0, len(indices), size)]
    if training and len(result) > 1 and len(result[-1]) == 1:
        result[-2].extend(result.pop())
    return result[:limit] if limit else result


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    import torch
    return dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def set_rng(state):
    import torch
    random.setstate(state['python']); torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def signature(config, encoder, graph, packets):
    trainer = {**TRAINER, **config.trainer}
    return dict(encoder=encoder, graph=graph, trainer=trainer, camera=config.camera,
        frame=config.frame, seed=config.seed, batch_size=config.batch_size, learning_rate=config.learning_rate,
        loss=config.loss, augmentation=config.augmentation, optimizer=config.optimizer, scheduler=config.scheduler,
        train_batch_limit=config.train_batch_limit, eval_batch_limit=config.eval_batch_limit,
        data={str(scene): digest(path/'manifest.json') for scene, path in packets})


def atomic_checkpoint(path, state):
    import os
    import torch
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def run(config, out):
    from grasppanda.compat import legacy_torch
    legacy_torch()
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from grasppanda.modules.gtg2 import GraphRegressor
    from grasppanda.optimization import build_optimizer, UpdateSchedule
    from grasppanda.module_options import unpack
    from grasppanda.losses import regression
    encoder, graph = resolved(config.modules)
    trainer = {**TRAINER, **config.trainer}
    packets = packet_paths(config, graph)
    native_aug = not config.augmentation or config.augmentation.get('mode') == 'native'
    dataset = GraphDataset(packets, graph, {'mode': 'none'} if native_aug else config.augmentation, config.seed)
    validation = GraphDataset(packets, graph, {'mode': 'none'}, config.seed)
    expected = signature(config, encoder, graph, packets)
    if not len(dataset): raise ValueError('Prepared GtG2 data contains no eligible candidate graphs')
    loss_kind, loss_options = unpack(config.loss.get('functions', {}).get('score', 'mse'))
    if loss_kind == 'upstream': loss_kind = 'mse'
    coefficient = config.loss.get('weights', {}).get('score', 1.)
    saved = None
    if config.checkpoint:
        saved = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        if saved.get('format') != FORMAT: raise ValueError('Expected a GraspPanda GtG2 ensemble checkpoint')
        if saved['signature']['encoder'] != encoder or saved['signature']['graph'] != graph:
            raise ValueError('GtG2 checkpoint encoder or graph contract differs')
        if config.train_checkpoint_mode == 'resume':
            if saved['signature'] != expected: raise ValueError('GtG2 resume training/data configuration differs')
            if config.scheduler and saved['horizon'] != config.epochs: raise ValueError('Resume must retain the original scheduled epoch horizon')
            if any(member['epoch'] > config.epochs for member in saved['members']): raise ValueError('Final epoch precedes saved training progress')
            if all(member['epoch'] >= config.epochs and not member.get('validation_pending', False) for member in saved['members']):
                raise ValueError('No epochs remain; increase the final epoch when using the native constant schedule')
    members = copy.deepcopy(saved['members']) if saved and config.train_checkpoint_mode == 'resume' else []
    losses = copy.deepcopy(saved.get('losses', [])) if saved and config.train_checkpoint_mode == 'resume' else []
    if not members:
        for fold in trainer['folds']:
            seed = (config.seed+fold*1009) % 2**32
            random.seed(seed); torch.manual_seed(seed)
            model = GraphRegressor(encoder, graph)
            if saved:
                matches = [m for m in saved['members'] if m['fold'] == fold and m['best'] is not None]
                if not matches: raise ValueError('Initialization checkpoint lacks a trained requested fold')
                model.load_state_dict(matches[0]['best'], strict=True)
            members.append(dict(fold=fold, epoch=0, model=cpu_state(model), best=None,
                best_loss=float('inf'), best_epoch=0, optimizer=None, scheduler=None, rng=rng_state(), validation_pending=False))
    if [m['fold'] for m in members] != trainer['folds']: raise ValueError('Checkpoint ensemble folds differ')
    out.mkdir(parents=True, exist_ok=True)
    training = out/'training'; training.mkdir(exist_ok=True)
    for member in members:
        fold = member['fold']
        val_indices = np.flatnonzero(validation.scenes % 10 == fold)
        if not len(val_indices): raise ValueError('A GtG2 fold has no held-out validation graphs')
        initial = training_indices(dataset, fold, 0, trainer, config.seed)
        updates = len(batches(initial, config.batch_size, config.train_batch_limit, training=True))
        model = GraphRegressor(encoder, graph).cuda()
        model.load_state_dict(member['model'], strict=True)
        optimizer = build_optimizer(model.parameters(), replace(config, optimizer=config.optimizer or {'type': 'adam'}))
        schedule = UpdateSchedule(optimizer, config.scheduler, updates*config.epochs) if config.scheduler else None
        if member['optimizer'] is not None: optimizer.load_state_dict(member['optimizer'])
        if schedule and member['scheduler'] is not None: schedule.load_state_dict(member['scheduler'])
        set_rng(member['rng'])

        def validate_member():
            model.eval(); total, count = 0., 0
            loader_options = dict(num_workers=config.data_workers,
                generator=torch.Generator().manual_seed(config.seed+member['epoch']-1))
            if config.data_workers: loader_options['multiprocessing_context'] = 'spawn'
            val_batches = batches(val_indices, config.batch_size, config.eval_batch_limit)
            with torch.no_grad():
                for batch in DataLoader(validation, batch_sampler=val_batches, **loader_options):
                    batch = batch.cuda()
                    prediction = model(batch)
                    if prediction.shape != batch.y.shape: raise ValueError('Graph validation target shape differs')
                    value = regression(prediction-batch.y, loss_kind, loss_options).mean()*coefficient
                    if not torch.isfinite(value): raise ValueError('Nonfinite GtG2 validation loss')
                    total += float(value)*batch.num_graphs; count += batch.num_graphs
            average = total/count
            if average < member['best_loss']:
                member.update(best_loss=average, best_epoch=member['epoch'], best=cpu_state(model))
            member.update(validation_pending=False, rng=rng_state())
            state = dict(format=FORMAT, signature=expected, horizon=config.epochs, members=members, losses=losses)
            atomic_checkpoint(training/f"fold_{fold}_epoch_{member['epoch']:04d}.pt", state)
            (out/'losses.json').write_text(json.dumps(losses, indent=2)+'\n')
            print(json.dumps(dict(fold=fold, epoch=member['epoch'], validation_loss=average)), flush=True)

        if member.get('validation_pending', False): validate_member()
        for epoch in range(member['epoch'], config.epochs):
            dataset.epoch, dataset.fold = epoch, fold
            indices = training_indices(dataset, fold, epoch, trainer, config.seed)
            order = batches(indices, config.batch_size, config.train_batch_limit,
                            seed=(config.seed+fold*1009+epoch*104729) % 2**32, training=True)
            if len(order) != updates: raise ValueError('Dynamic graph sampling changed the update horizon')
            loader_options = dict(num_workers=config.data_workers, generator=torch.Generator().manual_seed(config.seed+epoch))
            if config.data_workers: loader_options['multiprocessing_context'] = 'spawn'
            model.train()
            for step, batch in enumerate(DataLoader(dataset, batch_sampler=order, **loader_options)):
                optimizer.zero_grad(set_to_none=True)
                batch = batch.cuda()
                prediction = model(batch, native_augmentation=native_aug)
                if prediction.shape != batch.y.shape: raise ValueError('Graph prediction/target batch shapes differ')
                loss = regression(prediction-batch.y, loss_kind, loss_options).mean()*coefficient
                if not torch.isfinite(loss): raise ValueError('Nonfinite GtG2 training loss')
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError('Nonfinite GtG2 gradient')
                optimizer.step()
                if schedule: schedule.step()
                row = dict(step=len(losses)+1, epoch=epoch+1, fold=fold, total=float(loss.detach()), stage=f'fold_{fold}')
                losses.append(row); print(json.dumps(row), flush=True)
            member.update(epoch=epoch+1, model=cpu_state(model), optimizer=cpu_tree(optimizer.state_dict()),
                          scheduler=schedule.state_dict() if schedule else None, rng=rng_state(), validation_pending=True)
            # A complete optimizer epoch is recoverable even if validation fails.
            state = dict(format=FORMAT, signature=expected, horizon=config.epochs, members=members, losses=losses)
            path = training/f'fold_{fold}_epoch_{epoch+1:04d}.pt'
            atomic_checkpoint(path, state)
            validate_member()
        del model, optimizer, schedule
    path = out/'checkpoint.pt'
    atomic_checkpoint(path, dict(format=FORMAT, signature=expected, horizon=config.epochs, members=members, losses=losses))
    return dict(stage='gtg2_epoch_training', method='gtg2', losses=losses, checkpoint='checkpoint.pt',
        checkpoint_sha256=digest(path), members=[dict(fold=m['fold'], epoch=m['epoch'], selected_epoch=m['best_epoch'], validation_loss=m['best_loss']) for m in members],
        ap=None, protocol='Scene-disjoint reconstructed GtG2 ensemble; validation is candidate-score regression, not benchmark AP.')
