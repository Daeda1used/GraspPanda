"""Native ZeroGrasp objectives and Lightning optimization over local shards."""
from contextlib import contextmanager
import hashlib
import io
import json
import random
from types import MethodType

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from ..integrations.zerograsp11b import TRAINING_FIELDS, iter_samples, selected_ranges, shard_path, member_hashes
from ..jobs import digest
from ..training.state import same_state


@contextmanager
def sample_seed(value):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(value)
    np.random.seed(value)
    torch.set_rng_state(torch.Generator().manual_seed(value).get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def decode_sample(key, raw):
    from PIL import Image
    decoded = {'__key__':key}
    for name, data in raw.items():
        if name.endswith('.json'):
            decoded[name] = json.loads(data)
        elif name.endswith('.npz'):
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                decoded[name] = {key:archive[key].copy() for key in archive.files}
        else:
            with Image.open(io.BytesIO(data)) as image:
                decoded[name] = image.convert('RGB').copy() if name == 'rgb.jpg' else np.asarray(image).copy()
    surface, grasps = decoded['spc.npz'], decoded['grasp.npz']['grasp_poses']
    if surface['spc'].ndim != 2 or surface['spc'].shape[1] != 7 or surface['obj_ids'].shape != (len(surface['spc']),1):
        raise ValueError(f'{key}: invalid surface-point or instance-label layout')
    if grasps.ndim != 2 or len(grasps) != len(surface['spc']) or grasps.shape[1] not in (9,10):
        raise ValueError(f'{key}: surface points and grasp supervision are not aligned')
    return decoded


def collate(samples):
    return (torch.stack([value[0] for value in samples]),
            [[torch.as_tensor(value[1][0])] for value in samples],
            torch.stack([torch.as_tensor(value[2]) for value in samples]),
            [value[3] for value in samples], [value[4] for value in samples],
            torch.stack([torch.as_tensor(value[5]) for value in samples]),
            torch.stack([torch.as_tensor(value[6]) for value in samples]),
            [value[7] for value in samples])


def passthrough(value):
    return value


class LocalBatches(Dataset):
    """Cache at most one compressed shard's selected records per worker.

    Each item is a complete batch from one shard. A final partial batch is kept;
    every selected observation is consumed once per unbounded epoch.
    """
    def __init__(self, config, native):
        self.config, self.native, self.epoch = config, native, 0
        self.ranges = selected_ranges(config)
        self.batches, self.by_shard = [], {}
        for shard, indices in self.ranges.items():
            self.by_shard[shard] = []
            for offset in range(0, len(indices), config.batch_size):
                self.by_shard[shard].append(len(self.batches))
                self.batches.append((shard, indices[offset:offset+config.batch_size]))
        self.cached_shard, self.cached = None, {}

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        from zerograsp.utils.dataset import make_sample_wrapper
        shard, indices = self.batches[index]
        if self.cached_shard != shard:
            self.cached = {}
            wanted = set(self.ranges[shard])
            for ordinal, sample in enumerate(iter_samples(shard_path(self.config.dataset_root, shard), TRAINING_FIELDS)):
                if ordinal in wanted:
                    self.cached[ordinal] = sample
                    wanted.remove(ordinal)
                if not wanted:
                    break
            if wanted:
                raise ValueError(f'Incomplete training shard {shard}: missing selected samples {sorted(wanted)}')
            self.cached_shard = shard
        batch, evidence = [], []
        for ordinal in indices:
            key, raw = self.cached[ordinal]
            decoded = decode_sample(key, raw)
            camera = np.asarray(decoded['camera.json']['cam_K'], dtype=np.float32).reshape(3,3)
            seed = int.from_bytes(hashlib.sha256(f'{self.config.seed}/{self.epoch}/{shard}/{ordinal}'.encode()).digest()[:4], 'little')
            with sample_seed(seed):
                try:
                    value = make_sample_wrapper(self.native, is_eval=False, K=camera)(decoded)
                except Exception as error:
                    raise ValueError(f'Native ZeroGrasp preprocessing failed at shard {shard}, sample {key}: {error}') from error
            if not torch.isfinite(value[3].points).all() or not torch.isfinite(value[4].points).all():
                raise ValueError(f'{key}: non-finite normalized training points')
            batch.append(value)
            evidence.append(dict(shard=shard, sample=ordinal, key=key, member_sha256=member_hashes(raw)))
        return {'native':collate(batch), 'evidence':evidence}


class ShardOrder(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def set_epoch(self, epoch):
        self.dataset.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        shards = list(self.dataset.by_shard)
        random.Random(self.dataset.config.seed+self.dataset.epoch).shuffle(shards)
        for shard in shards:
            yield from self.dataset.by_shard[shard]


def signature(config):
    fields = config.to_dict()
    for name in ('checkpoint','checkpoint_policy','epochs','train_checkpoint_mode','timeout_minutes','gpu'):
        fields.pop(name, None)
    files = []
    for shard in selected_ranges(config):
        path = shard_path(config.dataset_root, shard)
        stat = path.stat()
        files.append(dict(file=path.name, bytes=stat.st_size, modified_ns=stat.st_mtime_ns))
    return {'config':fields, 'shards':files}


def run(config, out, steps=None):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from .zerograsp import build
    model, native, payload = build(config, out)
    expected = signature(config)
    resume = config.train_checkpoint_mode == 'resume'
    previous = payload.get('grasppanda', {})
    if resume:
        if previous.get('signature') != expected or not previous.get('resume_supported'):
            raise ValueError('Resume requires an epoch checkpoint with unchanged modules, selected data and training settings')
        if config.epochs <= int(payload.get('epoch', -1))+1:
            raise ValueError('Set epochs above the completed epoch count when resuming')
        if not payload.get('optimizer_states') or not payload.get('lr_schedulers'):
            raise ValueError('Resume checkpoint lacks native optimizer or scheduler state')
    dataset = LocalBatches(config, native)
    loader = DataLoader(dataset, batch_size=None, sampler=ShardOrder(dataset), collate_fn=passthrough,
        num_workers=config.data_workers, persistent_workers=False,
        generator=torch.Generator().manual_seed(config.seed))
    losses = []
    input_chain = [previous.get('input_chain','') if resume else '']
    original_forward = model.forward

    def training_step(self, batch, batch_index):
        terms, stats = original_forward(batch['native'])
        total = sum(terms.values())
        if not torch.isfinite(total):
            raise ValueError('Native ZeroGrasp returned a non-finite training loss')
        for name, value in {**terms, **stats, 'loss':total}.items():
            self.log('train_'+name, value, on_step=True, on_epoch=False,
                     batch_size=len(batch['native'][7]), logger=False)
        self.grasppanda_inputs = batch['evidence']
        return total

    def on_save(self, checkpoint):
        checkpoint['grasppanda'] = dict(signature=expected, config=config.to_dict(),
            resume_supported=steps is None, input_chain=input_chain[0])

    model.training_step = MethodType(training_step, model)
    model.on_save_checkpoint = MethodType(on_save, model)

    class Progress(pl.Callback):
        def on_train_start(self, trainer, module):
            if resume:
                checks = dict(model=same_state(module.state_dict(),payload['state_dict']),
                    optimizer=same_state(trainer.optimizers[0].state_dict(),payload['optimizer_states'][0]),
                    scheduler=same_state(trainer.lr_scheduler_configs[0].scheduler.state_dict(),payload['lr_schedulers'][0]),
                    step=trainer.global_step == payload['global_step'])
                if not all(checks.values()):
                    raise ValueError(f'ZeroGrasp checkpoint state did not restore exactly before training: {checks}')
                (out/'resume_restoration.json').write_text(json.dumps(dict(checks=checks,
                    optimizer_steps_start=trainer.global_step),indent=2)+'\n')

        def on_train_epoch_start(self, trainer, module):
            # Restore the same epoch seed for stochastic CVAE samples. Native
            # CUDA scatter/reduction kernels can still be numerically
            # nondeterministic; exact state restoration is checked separately.
            torch.manual_seed(config.seed+trainer.current_epoch)
            torch.cuda.manual_seed_all(config.seed+trainer.current_epoch)

        def on_before_optimizer_step(self, trainer, module, optimizer):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                raise ValueError('Native ZeroGrasp returned non-finite gradients')

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_index):
            record = dict(epoch=trainer.current_epoch, step=trainer.global_step,
                          total=float(outputs['loss'].detach().cpu()))
            record.update({name.removeprefix('train_'):float(value.detach().cpu())
                           for name,value in trainer.callback_metrics.items()
                           if name.startswith('train_loss_')})
            input_chain[0] = hashlib.sha256((input_chain[0]+json.dumps(module.grasppanda_inputs,sort_keys=True)).encode()).hexdigest()
            record['input_chain'] = input_chain[0]
            losses.append(record)
            if len(losses) > 1000:
                del losses[0]
            with (out/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            temporary = out/'result.json.tmp'
            temporary.write_text(json.dumps(dict(stage='real_label_training' if steps else 'native_epoch_training',
                dataset=config.dataset, method=config.method, losses=losses, optimizer_steps=trainer.global_step)))
            temporary.replace(out/'result.json')
            print(json.dumps(record), flush=True)

    class EpochCheckpoint(ModelCheckpoint):
        FILE_EXTENSION = '.pt'

    checkpoint_callback = EpochCheckpoint(dirpath=out, filename='checkpoint', auto_insert_metric_name=False,
        enable_version_counter=False, save_top_k=1, every_n_epochs=1, save_on_train_epoch_end=True)
    trainer = pl.Trainer(default_root_dir=out, accelerator='gpu', devices=1, precision='32-true',
        max_epochs=-1 if steps else config.epochs, max_steps=steps or -1,
        limit_train_batches=config.train_batch_limit or 1.0, limit_val_batches=0, num_sanity_val_steps=0,
        gradient_clip_val=.5, logger=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[Progress(), checkpoint_callback], log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=loader, ckpt_path=config.checkpoint if resume else None)
    if not losses:
        raise ValueError('ZeroGrasp training completed without an optimizer update')
    checkpoint_path = out/'checkpoint.pt'
    if not checkpoint_path.is_file():
        raise ValueError('Native training did not save its checkpoint')
    return dict(stage='real_label_training' if steps else 'native_epoch_training',
        dataset=config.dataset, method=config.method, camera=config.camera, losses=losses,
        optimizer_steps=trainer.global_step, epochs_completed=None if steps else config.epochs,
        selected_samples=config.frames, checkpoint_sha256=digest(checkpoint_path), input_chain=input_chain[0], ap=None,
        protocol_note='Native ZeroGrasp surface/grasp objectives, synthetic-depth noise, mask augmentation, AdamW, '
                      'step schedule and gradient clipping at 0.5. Epochs stream the selected local observations; '
                      'shards are shuffled, incomplete final batches are retained. No S3 upload or experiment account is required. '
                      'Per-sample seeds and scene-aware OFE are toolbox compatibility adaptations; no held-out AP is claimed.')
