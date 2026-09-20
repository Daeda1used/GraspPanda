"""Pinned GC6D Contact-GraspNet with its native heads and camera-frame decoder."""
import ast
import json
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

from ..config import ROOT, catalogue
from ..jobs import digest


def prepare():
    from ..worker import prepare as prepare_source
    import sys
    repo = prepare_source('contact_graspnet_gc6d')
    # utils is an upstream package, not a directory of independent imports.
    sys.path.remove(str(repo/'utils'))
    return repo


def configure(model, selection, voxel_size=.005):
    """Reuse registered 256-channel encoders, retaining the native grasp heads."""
    from torch import nn
    from ..components import configure_model, validate_selection
    from ..module_options import unpack
    validate_selection('contact_graspnet_gc6d', selection)
    choice, _ = unpack(selection.get('backbone', 'upstream'))
    if choice == 'upstream':
        return []
    proxy = nn.Module()
    proxy.view_estimator = nn.Module()
    proxy.view_estimator.backbone = nn.Identity()
    configure_model(proxy, 'graspnet_baseline', selection, voxel_size)
    model.backbone = proxy.view_estimator.backbone
    names = ('sa1_module', 'sa2_module', 'sa3_module', 'sa4_module',
             'fp1_module', 'fp2_module', 'fp3_module')
    for name in names:
        setattr(model, name, nn.Identity())
    # Execute the pinned author's decoder verbatim, after the encoder boundary.
    # This keeps contact-frame conventions, width bins and loss outputs native.
    path = ROOT/catalogue()['contact_graspnet_gc6d']['path']/'models/cgnet.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ContactGraspNet')
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    boundary = next(i for i, node in enumerate(forward.body) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'grasp_dir_head' for t in node.targets))
    function = ast.FunctionDef(name='_decode_grasppanda',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in ('self', 'l0_points', 'pred_points')],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=forward.body[boundary:], decorator_list=[])
    namespace = dict(model.forward.__func__.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    decode = namespace['_decode_grasppanda']
    def composed(self, points):
        features, seeds, _ = self.backbone(points)
        return decode(self, features, seeds.transpose(1, 2).contiguous())
    model.forward = MethodType(composed, model)
    return [name+'.' for name in (*names, 'backbone')]


def build(config, out=None):
    import torch
    from ..components import load_checkpoint
    prepare()
    from models.cgnet import ContactGraspNet
    model = ContactGraspNet(SimpleNamespace())
    prefixes = configure(model, config.modules, config.voxel_size)
    transfer = {'policy': 'random_initialization', 'initialized': sorted(model.state_dict()), 'discarded': []}
    checkpoint = {}
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint.get('base_model', checkpoint))
        transfer = load_checkpoint(model, state, prefixes, config.checkpoint_policy)
    if out is not None:
        (out/'component_transfer.json').write_text(json.dumps(transfer, indent=2)+'\n')
    return model.cuda(), checkpoint


def decode(prediction, mean):
    """The author's inference_single conversion, including width scaling/cap."""
    import numpy as np
    import torch
    poses = prediction['pred_grasps'].detach().cpu().clone()
    poses[:, :, :3, 3] += mean.cpu().reshape(1, 1, 3)
    scores = prediction['pred_scores'].detach().cpu().reshape(-1)
    values, indices = torch.topk(scores, k=min(2048, scores.numel()), largest=True)
    widths = prediction['pred_width'].detach().cpu()[0, indices, 0].numpy()
    poses = poses[0, indices].double().numpy()
    n = len(indices)
    return np.column_stack((values.numpy(), np.minimum(widths*1.2, .14),
                            np.full(n, .02), np.full(n, .02), poses[:, :3, :3].reshape(n, 9),
                            poses[:, :3, 3], np.full(n, -1)))


def infer(config, out):
    import numpy as np
    import torch
    from graspnetAPI import GraspGroup
    from ..datasets import get_dataset
    from ..integrations.graspclutter6d import frame_cloud, prediction_path
    from ..worker import overlay
    model, _ = build(config, out)
    model.eval()
    from utils.collision_detector import ModelFreeCollisionDetectorGPU
    records = []
    for index, (scene, frame) in enumerate(get_dataset(config.dataset).frame_keys(
            config.split, config.scene, config.frame, config.frames)):
        raw, xyz, intrinsics, rgb, evidence = frame_cloud(config, scene, frame)
        points = torch.from_numpy(xyz).float()
        mean = points.mean(0, keepdim=True)
        inputs = (points-mean).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            array = decode(model(inputs), mean)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        if array.ndim != 2 or array.shape[1] != 17 or not np.isfinite(array).all():
            raise ValueError('The native Contact-GraspNet decoder produced invalid grasps')
        grasps = GraspGroup(array)
        before = len(grasps)
        if config.collision_thresh > 0 and before:
            detector = ModelFreeCollisionDetectorGPU(torch.from_numpy(raw).cuda(), voxel_size=.01)
            mask = detector.detect(grasps, approach_dist=.05, collision_thresh=config.collision_thresh)
            grasps = grasps[~mask]
        path = out/'predictions'/prediction_path(scene, config.camera, frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        grasps.save_npy(str(path))
        if index == 0:
            overlay(rgb, grasps.grasp_group_array, intrinsics, out/'preview.png')
        record = dict(scene=scene, frame=frame, raw_points=len(raw), sampled_points=len(xyz),
                      grasps_before_collision=before, grasps_after_collision=len(grasps),
                      forward_decode_seconds=elapsed, prediction=str(path.relative_to(out)),
                      prediction_sha256=digest(path), **evidence)
        records.append(record)
        print(json.dumps(record), flush=True)
    sha = digest(config.checkpoint)
    manifest = {'config': config.to_dict(), 'checkpoint_sha256': sha,
                'files': {str(Path(r['prediction']).relative_to('predictions')): r['prediction_sha256'] for r in records}}
    (out/'predictions/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return dict(stage='dataset_inference', dataset=config.dataset, method=config.method,
                camera=config.camera, split=config.split, workspace=config.workspace, frames=records,
                checkpoint_sha256=sha, torch=torch.__version__, gpu=torch.cuda.get_device_name(), ap=None,
                protocol_note='Native GC6D Contact-GraspNet centering, heads, width conversion and collision filter. '
                              'The registered checkpoint was trained on realsense-d435; other cameras test cross-camera transfer.')
