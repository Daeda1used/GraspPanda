"""Native GC6D contact targets, losses and resumable epoch training."""
import json
from pathlib import Path
import random
from types import SimpleNamespace


def target_paths(config):
    from ..datasets import get_dataset
    from ..integrations.graspclutter6d import image_id
    spec = get_dataset(config.dataset)
    folder = Path(config.dataset_root)/'scene_contacts/train'
    if config.split != 'train' or config.workspace != 'official_gt_workspace':
        raise ValueError('Native GC6D contact training requires the train split and official_gt_workspace targets')
    if config.label_root:
        raise ValueError('GC6D contact targets use dataset_root/scene_contacts/train; label_root must be empty')
    if config.num_points != 20000:
        raise ValueError('The native GC6D training target protocol samples 20000 points')
    if config.eval_batch_limit:
        raise ValueError('The official GC6D contact trainer has no validation loop; evaluate test predictions separately')
    if config.action == 'train_short':
        pairs = [(config.scene, config.frame)]
    elif config.train_batch_limit:
        pairs = [(config.scene, frame) for frame in range(spec.frames_per_scene)
                 if (folder/f'{config.scene:06d}_{image_id(config.camera, frame):06d}.h5').is_file()]
        if not pairs:
            raise ValueError('No prepared GC6D contacts for the selected training scene; run ./panda prepare-contacts')
    else:
        pairs = [(scene, frame) for scene in spec.scene_ids('train') for frame in range(spec.frames_per_scene)]
    paths = [folder/f'{scene:06d}_{image_id(config.camera, frame):06d}.h5' for scene, frame in pairs]
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Missing contact targets: {path}. Run ./panda prepare-contacts; full training requires the published training split.')
    if config.action == 'train' and len(paths) < config.batch_size:
        raise ValueError('The native drop-last loader needs at least batch_size prepared samples; prepare more views or reduce batch_size')
    return paths


def seed_worker(_):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def run(config, out, steps=None):
    import numpy as np
    import torch
    from ..jobs import digest
    from .contact_gc6d import build
    paths = target_paths(config)
    model, payload = build(config, out)
    from data.gc6d import GraspClutter6D
    dataset = GraspClutter6D(SimpleNamespace(DATA_PATH=config.dataset_root,
        NUM_POSITIVE_CONTACTS=4000, CAMERA=config.camera, subset='train'))
    dataset.paths = [str(path) for path in paths]
    dataset.size = len(paths)
    # Preserve the native target construction/centering and positive-contact sampling.
    inputs = {str(path.relative_to(config.dataset_root)): digest(path) for path in paths}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240, eta_min=.00001)
    first_epoch, completed_updates, losses = 0, 0, []
    if config.optimizer or config.scheduler:
        raise ValueError('This adapter currently preserves native AdamW and the 240-epoch cosine schedule')
    if config.train_checkpoint_mode == 'resume':
        saved = payload.get('config', {})
        for key in ('dataset', 'method', 'dataset_root', 'camera', 'modules', 'workspace',
                    'num_points', 'batch_size', 'learning_rate', 'seed', 'data_workers', 'train_batch_limit'):
            if saved.get(key) != getattr(config, key):
                raise ValueError('Resume configuration differs: '+key)
        if payload.get('training_inputs') != inputs:
            raise ValueError('Prepared contact targets changed since the saved checkpoint')
        required = ('optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'rng')
        if any(key not in payload for key in required):
            raise ValueError('Resume requires a GraspPanda epoch checkpoint with optimizer, schedule and RNG state')
        first_epoch = payload['epoch']+1
        if first_epoch >= config.epochs:
            raise ValueError('Final epoch must exceed the completed checkpoint epoch')
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        completed_updates = payload['optimizer_steps']
        rng = payload['rng']
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state_all(rng['cuda'])
        np.random.set_state((rng['numpy_name'], rng['numpy_keys'].numpy().astype(np.uint32),
                             rng['numpy_position'], rng['numpy_gauss'], rng['numpy_cached']))
        random.setstate(rng['python'])

    def step(batch, epoch):
        nonlocal completed_updates
        batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['pc'])
        total, confidence, width, adds = model.get_loss(prediction, batch)
        if not torch.isfinite(total):
            raise ValueError('Native contact loss is non-finite; inspect positive-contact coverage of this batch')
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        completed_updates += 1
        record = dict(epoch=epoch+1, update=completed_updates, total=float(total.detach()),
                      confidence=float(confidence.detach()), width=float(width.detach()),
                      adds=float(adds.detach()), gradient_norm=float(norm), learning_rate=rate)
        losses.append(record)
        print(json.dumps(record), flush=True)
        progress = out/'result.json.tmp'
        progress.write_text(json.dumps(dict(stage='real_label_training' if steps else 'native_epoch_training', losses=losses)))
        progress.replace(out/'result.json')

    def save(epoch):
        state = np.random.get_state()
        record = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      scheduler_state_dict=scheduler.state_dict(), epoch=epoch, optimizer_steps=completed_updates,
                      config=config.to_dict(), training_inputs=inputs,
                      rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                               numpy_name=state[0], numpy_keys=torch.from_numpy(state[1].astype(np.int64)),
                               numpy_position=state[2], numpy_gauss=state[3], numpy_cached=state[4], python=random.getstate()))
        temporary = out/'checkpoint.pt.tmp'
        torch.save(record, temporary)
        temporary.replace(out/'checkpoint.pt')

    model.train()
    if steps is not None:
        # A short run deliberately repeats the selected annotated view.
        for index in range(steps):
            batch = torch.utils.data.default_collate([dataset[0] for _ in range(config.batch_size)])
            step(batch, 0)
        # Short checkpoints initialize prediction/training; they cannot resume epochs.
        torch.save(dict(model_state_dict=model.state_dict(), config=config.to_dict()), out/'checkpoint.pt')
    else:
        for epoch in range(first_epoch, config.epochs):
            loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                drop_last=True, num_workers=config.data_workers, worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(config.seed+epoch))
            for index, batch in enumerate(loader):
                step(batch, epoch)
                if config.train_batch_limit and index+1 >= config.train_batch_limit:
                    break
            scheduler.step()
            save(epoch)
    return dict(stage='real_label_training' if steps else 'native_epoch_training', dataset=config.dataset,
                method=config.method, camera=config.camera, losses=losses, optimizer_steps=completed_updates,
                epochs_completed=None if steps else config.epochs, checkpoint_sha256=digest(out/'checkpoint.pt'),
                training_inputs=inputs, ap=None,
                protocol_note='Author contact labels, 4000 positive-contact samples, native loss, AdamW, '
                              'gradient clipping at 1 and cosine schedule with T_max=240. Epoch training drops incomplete batches.')
