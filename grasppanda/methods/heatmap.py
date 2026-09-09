"""Headless adapters calling the original HGGD / RegionNormalizedGrasp demo pipeline.

The upstream depth clipping, camera constants, grouping, decoder and collision
detector are retained. These runs use a separate native_demo protocol.
"""
import importlib
import json
from pathlib import Path
import sys
import time

from grasppanda.jobs import digest


def infer(config, out):
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from graspnetAPI import GraspGroup
    from grasppanda.worker import prepare, overlay
    from grasppanda.components import configure_model, load_checkpoint

    prepare(config.method)
    camera_config = importlib.import_module('dataset.config')
    original_intrinsic = camera_config.get_camera_intrinsic
    camera_config.camera = config.camera
    camera_config.get_camera_intrinsic = lambda camera=config.camera: original_intrinsic(camera)
    sys.argv = ['demo.py', '--center-num', '48', '--all-points-num', str(config.num_points),
                '--group-num', '512', '--input-w', '640', '--input-h', '360']
    rng = config.method == 'region_normalized_grasp'
    sys.argv += ['--embed-dim', '256', '--patch-size', '64'] if rng else ['--anchor-num', '7']
    demo = importlib.import_module('demo')
    print('stage: construct networks', flush=True)
    demo.anchornet = demo.AnchorGraspNet(in_dim=4, ratio=8, anchor_k=6)
    changed = configure_model(demo.anchornet, config.method, config.modules)
    demo.anchornet.cuda().eval()
    demo.localnet = (demo.PatchMultiGraspNet(49, theta_k_cls=6, feat_dim=256, anchor_w=60)
                     if rng else demo.PointMultiGraspNet(info_size=3, k_cls=49)).cuda().eval()
    state = torch.load(config.checkpoint, map_location='cuda', weights_only=True)
    print('stage: load checkpoint', flush=True)
    transfer = load_checkpoint(demo.anchornet, state['anchor'], changed, config.checkpoint_policy)
    # Author checkpoints contain THOP profiling counters. They are not learned
    # parameters; remove only these counters, then require an exact model match.
    local_state = {k:v for k,v in state['local'].items() if k.rsplit('.',1)[-1] not in ('total_ops','total_params')}
    demo.localnet.load_state_dict(local_state, strict=True)
    demo.anchors = {k: state[k].cuda() for k in ('gamma', 'beta')}
    helper = demo.PointCloudHelper(config.num_points)
    print('stage: read frame', flush=True)
    collision = demo.collision_detect
    counts = {}

    def measured_collision(points, grasps, **kwargs):
        counts['grasps_before_collision'] = len(grasps)
        result, mask = collision(points, grasps, **kwargs)
        counts['grasps_after_collision'] = len(result)
        return result, mask

    demo.collision_detect = measured_collision
    records = []
    for index in range(config.scene * 256 + config.frame, config.scene * 256 + config.frame + config.frames):
        scene, frame = divmod(index, 256)
        directory = Path(config.dataset_root) / 'scenes' / f'scene_{scene:04d}' / config.camera
        rgb_path = directory / 'rgb' / f'{frame:04d}.png'
        depth_path = directory / 'depth' / f'{frame:04d}.png'
        depth_array = np.clip(np.asarray(Image.open(depth_path)), 0, 1000).astype(np.float32)
        if np.count_nonzero(depth_array) < config.num_points:
            raise ValueError('Upstream heatmap sampler requires at least num_points valid depth pixels; lower num_points.')
        rgb = torch.from_numpy(np.asarray(Image.open(rgb_path)).copy()).cuda().float().permute(2, 1, 0)[None] / 255.
        depth = torch.from_numpy(depth_array.T.copy()).cuda()[None]
        torch.cuda.synchronize()
        start = time.monotonic()
        view, _, _ = helper.to_scene_points(rgb, depth, include_rgb=True)
        xyz = helper.to_xyz_maps(depth)
        small_depth = F.interpolate(depth[None], (640, 360))[0] / 1000.
        small_depth = torch.clip(small_depth - small_depth.mean(), -1, 1)
        x = torch.cat([small_depth[None], F.interpolate(rgb, (640, 360))], 1)
        counts.clear()
        with torch.no_grad():
            print('stage: upstream inference', flush=True)
            if rng:
                gg = demo.inference(view.squeeze(0), torch.cat([rgb.squeeze(0), xyz.squeeze(0)], 0),
                                    x, rgb, depth, use_heatmap=True, vis_heatmap=False, vis_grasp=False)
            else:
                gg = demo.inference(view, xyz, x, depth, vis_heatmap=False, vis_grasp=False)
        torch.cuda.synchronize()
        rows = [] if gg is None else [[g.score, g.width, g.height, g.depth,
                                      *g.rotation.reshape(-1), *g.translation, -1] for g in gg]
        array = np.asarray(rows, dtype=np.float64).reshape(-1, 17)
        if not np.isfinite(array).all():
            raise ValueError('Non-finite output from upstream heatmap pipeline')
        target = out / 'predictions' / f'scene_{scene:04d}' / config.camera / f'{frame:04d}.npy'
        target.parent.mkdir(parents=True, exist_ok=True)
        GraspGroup(array).save_npy(str(target))
        if not records:
            overlay(rgb_path, array, camera_config.get_camera_intrinsic(), out / 'preview.png')
        row = dict(scene=scene, frame=frame, sampled_points=config.num_points,
                   grasps_saved=len(array), **counts, pipeline_seconds=time.monotonic()-start,
                   prediction=str(target.relative_to(out)), prediction_sha256=digest(target),
                   depth_sha256=digest(depth_path), rgb_sha256=digest(rgb_path))
        records.append(row)
        print(json.dumps(row), flush=True)
    manifest = dict(config=config.to_dict(), checkpoint_sha256=digest(config.checkpoint),
                    files={str(Path(r['prediction']).relative_to('predictions')):r['prediction_sha256'] for r in records})
    (out/'predictions/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    faulthandler.cancel_dump_traceback_later()
    return dict(stage='dataset_inference', method=config.method, camera=config.camera, split=config.split,
                modules=config.modules, checkpoint_transfer=transfer,
                workspace=config.workspace, frames=records, checkpoint_sha256=digest(config.checkpoint),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(), ap=None,
                protocol_note='Original demo: RGB-D, fixed author camera intrinsics, depth clipped to 1 m, '
                'no GT mask; native collision and NMS. RNG also retains score > 0.5 and top 50. '
                'This is a demo protocol, not a claimed reproduction of full benchmark AP.',
                timing_note='End-to-end first-call pipeline including postprocessing; not warmed FPS.')
