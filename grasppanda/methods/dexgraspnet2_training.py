"""Native hand losses with finite selected-view epochs and complete resume state."""
import json
from pathlib import Path
import random


def run(config, out, steps=None):
    import numpy as np
    import torch
    from ..integrations.dexgraspnet2 import TrainingViews, collate
    from ..jobs import digest
    from .dexgraspnet2 import build
    model, native, payload = build(config, out)
    dataset = TrainingViews(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=native.max_iter, eta_min=native.lr_min)
    first_epoch, updates, losses, inputs = 0, 0, [], {}
    signature = {key:getattr(config,key) for key in ('dataset','method','dataset_root','camera','split','scene',
        'frame','frames','modules','workspace','voxel_size','num_points','batch_size','learning_rate','seed',
        'data_workers','train_batch_limit')}
    if config.train_checkpoint_mode == 'resume':
        saved = payload.get('grasppanda', {})
        if not saved.get('resume_supported') or saved.get('signature') != signature:
            raise ValueError('Resume requires a matching GraspPanda epoch checkpoint and unchanged selected views/configuration.')
        inputs = saved['inputs']
        for relative, expected in inputs.items():
            path = Path(config.dataset_root)/relative
            if not path.is_file() or digest(path) != expected:
                raise ValueError('A training input changed since the checkpoint: '+relative)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        from ..training.state import same_state
        checks = dict(model=same_state(model.state_dict(),payload['model']),
                      optimizer=same_state(optimizer.state_dict(),payload['optimizer']),
                      scheduler=same_state(scheduler.state_dict(),payload['scheduler']))
        if not all(checks.values()):
            raise ValueError('Native hand checkpoint state did not restore exactly: '+str(checks))
        (out/'resume_restoration.json').write_text(json.dumps(checks,indent=2)+'\n')
        first_epoch, updates = payload['epoch']+1, payload['optimizer_steps']
        if first_epoch >= config.epochs:
            raise ValueError('Final epoch must exceed the completed checkpoint epoch.')
        rng = payload['rng']
        random.setstate(rng['python']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
        np.random.set_state((rng['numpy_name'],rng['numpy_keys'].numpy().astype(np.uint32),
                             rng['numpy_position'],rng['numpy_gauss'],rng['numpy_cached']))

    def save(epoch, resume_supported):
        state = np.random.get_state()
        record = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            optimizer_steps=updates, epoch=epoch, grasppanda=dict(config=config.to_dict(),signature=signature,
            inputs=inputs,resume_supported=resume_supported),
            rng=dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                     numpy_name=state[0],numpy_keys=torch.from_numpy(state[1].astype(np.int64)),
                     numpy_position=state[2],numpy_gauss=state[3],numpy_cached=state[4]))
        temporary = out/'checkpoint.pt.tmp'; torch.save(record,temporary); temporary.replace(out/'checkpoint.pt')

    def step(batch, epoch):
        nonlocal updates
        tensors, evidence = batch
        tensors = {key:value.cuda(non_blocking=True) for key,value in tensors.items()}
        optimizer.zero_grad(set_to_none=True)
        total, terms = model(tensors)
        if not torch.isfinite(total):
            raise ValueError('The native dexterous objective returned a non-finite loss.')
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), native.grad_clip, error_if_nonfinite=True)
        rate = optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step(); updates += 1
        for files in evidence:
            for relative, sha in files.items():
                if relative in inputs and inputs[relative] != sha:
                    raise ValueError('Training input changed during the run: '+relative)
                inputs[relative] = sha
        record = dict(epoch=epoch+1,update=updates,total=float(total.detach()),gradient_norm=float(gradient_norm),
                      learning_rate=rate,**{key:float(value.detach().mean()) for key,value in terms.items()})
        losses.append(record)
        if len(losses)>1000: del losses[0]
        with (out/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        progress = out/'result.json.tmp'
        progress.write_text(json.dumps(dict(stage='real_label_training' if steps else 'native_epoch_training',losses=losses)))
        progress.replace(out/'result.json'); print(json.dumps(record),flush=True)

    model.train()
    for epoch in range(first_epoch, config.epochs if steps is None else steps):
        dataset.epoch = epoch
        loader = torch.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=True,drop_last=False,
            num_workers=config.data_workers,collate_fn=collate,generator=torch.Generator().manual_seed(config.seed+epoch))
        for index,batch in enumerate(loader):
            step(batch,epoch)
            if steps is not None and updates>=steps: break
            if config.train_batch_limit and index+1>=config.train_batch_limit: break
        if steps is not None and updates>=steps:
            save(epoch,False); break
        if steps is None: save(epoch,True)
    return dict(stage='real_label_training' if steps else 'native_epoch_training',dataset=config.dataset,
        method=config.method,camera=config.camera,losses=losses,optimizer_steps=updates,
        epochs_completed=None if steps else config.epochs,selected_views=len(dataset),
        checkpoint_sha256=digest(out/'checkpoint.pt'),ap=None,simulation_success=None,
        protocol_note='Native object/graspness and hand objectives, equal-object label sampling, 128 proposals, '
            '64 matched grasp seeds, 6 mm matching, in-plane rotation, Adam, gradient clipping at 10 '
            'and the 50000-update cosine schedule. Toolbox epochs visit each selected view once and retain '
            'the final partial batch. Checkpoints retain the optimizer, scheduler and random states.')
