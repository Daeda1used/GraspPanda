"""Bounded native GraspFast training with its released unweighted objective."""
import ast
import importlib
import sys
import time
import types
from pathlib import Path


def run(config, out, steps=3):
    import numpy as np
    import scipy.io
    import torch
    from grasppanda.methods.graspfast import prepare, checkpoint_state, guard
    from grasppanda.config import ROOT, catalogue
    from grasppanda.jobs import digest
    module = prepare()
    repo = ROOT / catalogue()['graspfast']['path']
    root = Path(config.dataset_root)
    stage = out / 'prepared'
    stage.mkdir(exist_ok=True)
    for name in ('scenes', 'grasp_label', 'collision_label'):
        (stage / name).symlink_to(root / name, target_is_directory=True)
    # Bound the native generator to one frame; preserve its target formulas.
    source = repo / 'DataProcessing/generate_scenes.py'
    tree = ast.parse(source.read_text())
    class FrameSelection(ast.NodeTransformer):
        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id in ('scene_id', 'ann_id'):
                value = config.scene if node.target.id == 'scene_id' else config.frame
                node.iter = ast.List(elts=[ast.Constant(value)], ctx=ast.Load())
            return self.generic_visit(node)
    utils = types.ModuleType('utils'); utils.__path__ = []
    sys.modules['utils'] = utils
    sys.modules['utils.data_utils'] = importlib.import_module('data_utils')
    argv = sys.argv
    try:
        sys.argv = [str(source), '--dataset_root', str(stage), '--camera_type', config.camera]
        exec(compile(ast.fix_missing_locations(FrameSelection().visit(tree)), str(source), 'exec'),
             {'__name__': '__main__', '__file__': str(source)})
    finally:
        sys.argv = argv
    target = stage / 'GraspFast' / f'scene_{config.scene:04d}' / config.camera / f'{config.frame:04d}.npy'
    if not np.isfinite(np.load(target)).all():
        raise ValueError('Native GraspFast target generation produced non-finite values')
    dataset_module = importlib.import_module('DataProcessing.graspfast_dataset')
    directory = root / 'scenes' / f'scene_{config.scene:04d}' / config.camera
    ids = scipy.io.loadmat(directory / 'meta' / f'{config.frame:04d}.mat')['cls_indexes'].reshape(-1)
    labels = {}; evidence = {}
    for obj in ids:
        path = root / 'grasp_label_simplified' / f'{int(obj)-1:03d}_labels.npz'
        with np.load(path) as data:
            labels[int(obj)] = tuple(data[k].astype(np.float32) for k in ('points', 'width', 'scores'))
        evidence[str(int(obj)-1)] = digest(path)
    dataset = dataset_module.GraspFastDataset(str(root), labels, camera=config.camera, split='train',
        num_points=config.num_points, voxel_size=config.voxel_size, remove_outlier=True, augment=False, load_label=False)
    scene = f'scene_{config.scene:04d}'
    collision = root / 'collision_label' / scene / 'collision_labels.npz'
    with np.load(collision) as data:
        dataset.collision_labels[scene] = {i: data[f'arr_{i}'] for i in range(len(data))}
    index = config.scene * 256 + config.frame
    dataset.graspfastpath[index] = str(target)
    dataset.load_label = True
    batch = dataset_module.minkowski_collate_fn([dataset[index]])
    def cuda(value):
        if isinstance(value, torch.Tensor): return value.cuda()
        if isinstance(value, dict): return {k: cuda(v) for k, v in value.items()}
        if isinstance(value, list): return [cuda(v) for v in value]
        return value
    batch = cuda(batch)
    model = module.GraspFast(is_training=True).cuda().train()
    model.load_state_dict(checkpoint_state(torch.load(config.checkpoint, map_location='cpu', weights_only=True)), strict=True)
    guard(model, module)
    get_loss = importlib.import_module('GraspFastModel.loss').get_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses = []; updates = []; start = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        end = model(dict(batch))
        end.update(objectness_score=end['object_criteria'], graspness_score=end['graspable_score'],
                   graspness_label=end['graspfast_label'])
        loss, end = get_loss(end)
        parts = {k: float(v.detach()) for k, v in end.items() if k.startswith('loss/')}
        if not torch.isfinite(loss) or not all(np.isfinite(v) for v in parts.values()):
            raise ValueError('Non-finite native GraspFast loss')
        loss.backward()
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Invalid GraspFast gradients')
        selected = next((p for p in parameters if torch.count_nonzero(p.grad)), None)
        if selected is None: raise ValueError('No nonzero GraspFast gradients')
        before = selected.detach().clone()
        norm = float(torch.sqrt(sum(p.grad.detach().square().sum() for p in parameters)))
        optimizer.step()
        delta = float((selected.detach()-before).norm())
        if not delta > 0 or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Invalid GraspFast parameter update')
        losses.append(dict(total=float(loss.detach()), components=parts))
        updates.append(dict(gradient_norm=norm, parameter_update_norm=delta))
        print('LOSS', losses[-1], flush=True)
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    config=config.to_dict()), out/'checkpoint.pt')
    return dict(method='graspfast', camera=config.camera, scene=config.scene, frame=config.frame,
        stage='native_unweighted_training', optimizer_steps=steps, losses=losses, updates=updates,
        seconds=time.monotonic()-start, target_sha256=digest(target), label_sha256=evidence,
        collision_sha256=digest(collision), checkpoint_sha256=digest(config.checkpoint), ap=None,
        protocol='Native generated graspability targets and released get_loss with five objectives. Strict author checkpoint after lossless module renames and linear head split; tensor-name aliases bridge released model/loss names. The separate weighted training driver remains unvalidated because its mask semantics are inconsistent.')
