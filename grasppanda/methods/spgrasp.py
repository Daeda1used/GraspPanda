"""Prompted planar sequence workflows using the pinned SPGrasp implementation."""
from copy import deepcopy
from pathlib import Path
import json
import time

from .spgrasp_options import PLANAR, TRAINER, architecture


def preflight(config):
    from PIL import Image
    from .spgrasp_data import frame_paths, rectangle_directory, Letterbox, prepare_prompts
    count = config.frames * (config.batch_size if config.action == 'train_check' else 1)
    paths = frame_paths(config.dataset_root, config.scene, config.camera, config.frame, count)
    if not config.checkpoint or not Path(config.checkpoint).is_file():
        raise ValueError('Select a trained SPGrasp checkpoint for inference, or download SAM2 initialization for training')
    if config.action == 'infer':
        with Image.open(paths[0]) as rgb:
            prepare_prompts(config.prompts, Letterbox(rgb.height, rgb.width))
    else:
        rect = rectangle_directory(config.dataset_root, config.scene, config.camera, config.label_root)
        for rgb in paths:
            for path in (rgb.parent.parent/'label'/rgb.name, rect/(rgb.stem+'.npy')):
                if not path.is_file():
                    raise ValueError('SPGrasp training target missing: '+str(path))


def _match_checkpoint(config, model, width_scale):
    from .spgrasp_model import Architecture
    if config.modules and Architecture(**architecture(config.modules)) != model._grasppanda_architecture:
        raise ValueError('SPGrasp component settings differ from the selected grasp checkpoint')
    if 'width_scale_pixels' in config.planar and config.planar['width_scale_pixels'] != width_scale:
        raise ValueError('Width normalization must match the trained SPGrasp checkpoint')


def infer(config, out):
    import torch
    from ..overlays import prepare_overlay
    from ..jobs import digest
    from .spgrasp_model import load_checkpoint, predict_sequence
    from .spgrasp_data import read_rgb_sequence, prepare_prompts, decode_maps, draw_preview
    start = time.monotonic()
    model, _, width_scale = load_checkpoint(prepare_overlay('spgrasp'), config.checkpoint, 'cuda')
    _match_checkpoint(config, model, width_scale)
    images, geometry, paths = read_rgb_sequence(config.dataset_root, config.scene, config.camera,
        config.frame, config.frames, model._grasppanda_architecture.resolution)
    ids, prompts = prepare_prompts(config.prompts, geometry)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = predict_sequence(model, images.cuda(), {k: v.cuda() for k, v in prompts.items()})
    options = {**PLANAR, **config.planar}
    options.pop('width_scale_pixels')
    rows = []
    for index, (maps, path) in enumerate(zip(logits, paths)):
        grasps = decode_maps(maps, geometry, ids, width_scale,
            width_positive_weight=model._grasppanda_width_positive_weight, **options)
        rows.append(dict(scene=config.scene, frame=config.frame+index, grasps=grasps,
                         grasps_saved=len(grasps), rgb_sha256=digest(path)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'planar-grasps.json').write_text(json.dumps(rows, indent=2)+'\n')
    draw_preview(paths[-1], rows[-1]['grasps'], out/'preview.png')
    return dict(stage='planar_sequence_inference', method='spgrasp', camera=config.camera,
        split=config.split, frames=rows, prompts=config.prompts, modules=config.modules,
        architecture=model._grasppanda_architecture.to_dict(), width_scale_pixels=width_scale,
        width_positive_weight=model._grasppanda_width_positive_weight,
        output_format='Original RGB pixel centers, opening angle in radians, opening width in pixels',
        protocol='RGB and explicit first-frame prompts; no annotation or depth input, no 3D collision or GraspNet AP',
        checkpoint_sha256=digest(config.checkpoint), seconds=time.monotonic()-start, ap=None)


def _training_clips(config, resolution, width_scale):
    """Build native video objects once; preserve continuous targets and geometry."""
    import random
    import torch
    from PIL import Image
    from training.utils.data_utils import VideoDatapoint, Frame, Object
    from .spgrasp_data import frame_paths, native_targets, Letterbox
    clips, selection = [], []
    settings = {**TRAINER, **config.trainer}
    for clip in range(config.batch_size):
        first = config.frame + clip*config.frames
        paths = frame_paths(config.dataset_root, config.scene, config.camera, first, config.frames)
        initial = native_targets(config.dataset_root, config.scene, config.camera, first, width_scale, label_root=config.label_root)
        eligible = [oid for oid, target in initial.items() if bool((target[0] > 0).any())]
        if not eligible:
            raise ValueError('The initial training frame has no visible object with a positive grasp target')
        ids = sorted(random.sample(eligible, min(settings['objects'], len(eligible))))
        frames, geometry = [], None
        for index, path in enumerate(paths):
            with Image.open(path) as rgb:
                image = rgb.convert('RGB')
            if geometry is None:
                geometry = Letterbox(image.height, image.width, resolution)
            targets = initial if index == 0 else native_targets(config.dataset_root, config.scene,
                config.camera, first+index, width_scale, object_ids=ids, label_root=config.label_root)
            objects = [Object(oid, first+index, geometry.targets(targets[oid])) for oid in ids]
            frames.append(Frame(geometry.image(image), objects))
        clips.append(VideoDatapoint(frames, config.scene*256+first, (geometry.height, geometry.width)))
        selection.append(dict(scene=config.scene, first_frame=first, frames=config.frames, objects=ids))
    return clips, selection


def _transforms(config, native_config):
    from hydra.utils import instantiate
    from training.dataset.transforms import ColorJitter, RandomGrayscale, ComposeAPI
    options = config.augmentation
    mode = options.get('mode', 'native')
    native = native_config.vos.train_transforms[0].transforms
    # Geometry was already transformed using the configured resolution. Keep the
    # released photometric transforms and normalization without its fixed resize.
    if mode == 'native':
        operations = [instantiate(t) for t in native[1:]]
    else:
        operations = []
        if mode == 'custom':
            consistent = options.get('consistent', True)
            operations += [ColorJitter(consistent, options.get('brightness', .1),
                options.get('contrast', .03), options.get('saturation', .03), None),
                RandomGrayscale(consistent, options.get('grayscale', .05))]
        operations += [instantiate(t) for t in native[-2:]]
    return ComposeAPI(operations)


def train(config, out, steps):
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from ..overlays import prepare_overlay
    from ..training.optimization import build_optimizer, UpdateSchedule
    from ..jobs import digest
    from .spgrasp_model import (Architecture, create_model, initialize_sam2, load_checkpoint,
        save_checkpoint, set_training, validate_width_calibration)
    start = time.monotonic()
    source = prepare_overlay('spgrasp')
    payload = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
    restored = isinstance(payload, dict) and payload.get('format') == 'grasppanda.spgrasp.v1'
    del payload
    if restored:
        model, native, width_scale = load_checkpoint(source, config.checkpoint)
        _match_checkpoint(config, model, width_scale)
        if model._grasppanda_width_positive_weight != 10:
            raise ValueError('Native width loss requires a checkpoint calibrated with positive weight ten')
        initialization = {'kind': 'trained SPGrasp checkpoint', 'policy': 'strict'}
    else:
        model, native = create_model(source, Architecture(**architecture(config.modules)))
        width_scale = config.planar.get('width_scale_pixels', PLANAR['width_scale_pixels'])
        initialization = initialize_sam2(model, config.checkpoint)
    settings = {**TRAINER, **config.trainer}
    model.prob_to_use_box_input_for_train = settings['box_probability']
    model.num_correction_pt_per_frame = settings['correction_clicks']
    model.num_init_cond_frames_for_train = settings['conditioning_frames']
    model.num_frames_to_correct_for_train = settings['correction_frames']
    model.rng = np.random.default_rng(config.seed)
    model.cuda()
    set_training(model, True)
    native.scratch.base_lr = config.learning_rate
    native.scratch.vision_lr = config.learning_rate*.6
    names = dict(position='loss_pos', angle='loss_ang', width='loss_wid', semantic='loss_semantic')
    for key, value in config.loss.get('weights', {}).items():
        native.trainer.loss.all.weight_dict[names[key]] = value
    loss_fn = instantiate(native.trainer.loss.all)
    OmegaConf.register_new_resolver('divide', lambda a, b: a/b, replace=True)
    from training.optimizer import construct_optimizer
    from training.utils.data_utils import collate_fn
    native_optimizer = None
    if config.optimizer:
        optimizer = build_optimizer((p for p in model.parameters() if p.requires_grad), config)
    else:
        native_optimizer = construct_optimizer(model, native.trainer.optim.optimizer,
            native.trainer.optim.options, native.trainer.optim.param_group_modifiers)
        optimizer = native_optimizer.optimizer
    schedule = UpdateSchedule(optimizer, config.scheduler or {'type': 'cosine', 'min_lr_ratio': .1}, steps) if config.scheduler or config.optimizer else None
    clips, selection = _training_clips(config, model._grasppanda_architecture.resolution, width_scale)
    transform = _transforms(config, native)
    losses = []
    for step in range(steps):
        batch = collate_fn([transform(deepcopy(clip)) for clip in clips], 'all').to('cuda')
        validate_width_calibration(batch.masks)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predictions = model(batch)
            values = loss_fn(predictions, batch.masks)
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError('SPGrasp produced a non-finite training objective')
        values['core_loss'].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise ValueError('SPGrasp training produced missing or non-finite gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        if schedule:
            optimizer.step()
            schedule.step()
        else:
            native_optimizer.step(where=step/steps, step=step)
        row = dict(step=step+1, stage='planar_training', total=float(values['core_loss'].detach()),
            **{k: float(v.detach()) for k, v in values.items()}, gradient_norm=float(norm))
        losses.append(row)
        print(json.dumps(row), flush=True)
        del batch, predictions, values, gradients
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out/'checkpoint.pt', width_scale,
        metadata=dict(config=config.to_dict(), selection=selection, updates=steps, width_objective='native capped weighted BCE'))
    return dict(stage='planar_training', method='spgrasp', losses=losses, updates=steps,
        checkpoint='checkpoint.pt', initialization=initialization, architecture=model._grasppanda_architecture.to_dict(),
        modules=config.modules, selection=selection, width_scale_pixels=width_scale, width_positive_weight=10,
        optimizer_groups=len(optimizer.param_groups), checkpoint_sha256=digest(out/'checkpoint.pt'),
        protocol='Bounded labelled clips with simulated point/box prompts, float32 planar targets and native losses; no convergence or AP claim',
        seconds=time.monotonic()-start, ap=None)
