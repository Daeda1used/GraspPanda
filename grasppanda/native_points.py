"""Sparse methods using their own original dataset loader and collator."""
import importlib
import json
from pathlib import Path
import sys
import time

from .jobs import digest


def infer(config, out):
    import numpy as np
    import scipy.io
    import torch
    from graspnetAPI import GraspGroup
    from .worker import prepare, overlay
    from .config import SPLITS

    prepare(config.method)
    sys.argv = ['test.py', '--dataset_root', config.dataset_root, '--camera', config.camera]
    economic = config.method == 'economicgrasp'
    mod = importlib.import_module('models.economicgrasp' if economic else 'graspnet')
    dataset_mod = importlib.import_module('dataset.graspnet_dataset')
    model = (mod.economicgrasp if economic else mod.GraspNet)(is_training=False).cuda().eval()
    state = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=True)
    threshold = mod.cfgs.graspness_threshold if economic else mod.GRASPNESS_THRESHOLD

    def guard(_module, _inputs, end):
        mask = (end['objectness_score'].argmax(1) == 1) & (end['graspness_score'].squeeze(1) > threshold)
        if (mask.sum(1) == 0).any():
            raise ValueError('No graspable points; upstream FPS requires a nonempty candidate set')

    model.graspable.register_forward_hook(guard)
    dataset = dataset_mod.GraspNetDataset(config.dataset_root, camera=config.camera, split=config.split,
                num_points=config.num_points, voxel_size=config.voxel_size,
                remove_outlier=True, augment=False, load_label=False)
    collate = dataset_mod.collate_fn if economic else dataset_mod.minkowski_collate_fn
    collision = importlib.import_module('utils.collision_detector' if economic else 'collision_detector').ModelFreeCollisionDetector
    records = []
    for index in range(config.scene*256+config.frame, config.scene*256+config.frame+config.frames):
        scene, frame = divmod(index, 256)
        data_index = index - SPLITS[config.split][0]*256
        sample = dataset[data_index]
        inputs = collate([sample])
        inputs = {k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        raw = dataset.get_data(data_index, return_raw_cloud=True)
        if isinstance(raw, tuple): raw = raw[0]
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.no_grad(): array = mod.pred_decode(model(inputs))[0].cpu().numpy()
        torch.cuda.synchronize(); elapsed = time.monotonic()-start
        if array.ndim != 2 or array.shape[1] != 17 or not np.isfinite(array).all():
            raise ValueError('Invalid native decoder output')
        gg = GraspGroup(array); before = len(gg)
        if config.collision_thresh > 0 and len(gg):
            # Same native detector; collision voxelization is fixed at 1 cm in this recipe.
            gg = gg[~collision(raw, voxel_size=0.01).detect(gg, approach_dist=0.05, collision_thresh=config.collision_thresh)]
        target = out/'predictions'/f'scene_{scene:04d}'/config.camera/f'{frame:04d}.npy'
        target.parent.mkdir(parents=True, exist_ok=True); gg.save_npy(str(target))
        directory = Path(config.dataset_root)/'scenes'/f'scene_{scene:04d}'/config.camera
        rgb = directory/'rgb'/f'{frame:04d}.png'; meta = directory/'meta'/f'{frame:04d}.mat'
        if not records: overlay(rgb, gg.grasp_group_array, scipy.io.loadmat(meta)['intrinsic_matrix'], out/'preview.png')
        evidence = {name+'_sha256':digest(path) for name,path in {
            'depth':directory/'depth'/f'{frame:04d}.png','rgb':rgb,'meta':meta,
            'label':directory/'label'/f'{frame:04d}.png','poses':directory/'camera_poses.npy',
            'table':directory/'cam0_wrt_table.npy'}.items()}
        row = dict(scene=scene, frame=frame, raw_points=len(raw), sampled_points=config.num_points,
                   grasps_before_collision=before, grasps_after_collision=len(gg),
                   forward_decode_seconds=elapsed, prediction=str(target.relative_to(out)),
                   prediction_sha256=digest(target), **evidence)
        records.append(row); print(json.dumps(row), flush=True)
    (out/'predictions/manifest.json').write_text(json.dumps(dict(config=config.to_dict(),
        checkpoint_sha256=digest(config.checkpoint), files={str(Path(r['prediction']).relative_to('predictions')):
        r['prediction_sha256'] for r in records}), indent=2)+'\n')
    return dict(stage='dataset_inference', method=config.method, camera=config.camera, split=config.split,
        workspace=config.workspace, frames=records, checkpoint_sha256=digest(config.checkpoint), ap=None,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        protocol_note='Original data loader, model, decoder and collision detector; GT workspace; collision voxel 1 cm.',
        timing_note='First-call forward + decode, not warmed FPS.')
