"""Candidate generation and mean-score inference with trained graph ensembles."""
import json
from pathlib import Path
import time

from grasppanda.config import ROOT, catalogue
from grasppanda.jobs import digest


def load_ensemble(config):
    import torch
    from grasppanda.methods.gtg2_options import resolved
    from grasppanda.methods.gtg2_training import FORMAT
    from grasppanda.modules.gtg2 import GraphRegressor
    encoder, graph = resolved(config.modules)
    state = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('GtG2 inference requires a trained GraspPanda ensemble checkpoint')
    model_graph = lambda value: {k: v for k, v in value.items() if k not in ('candidate_limit', 'gpg_threads')}
    if state['signature']['encoder'] != encoder or model_graph(state['signature']['graph']) != model_graph(graph):
        raise ValueError('Checkpoint graph or encoder configuration differs; retain the training module settings')
    members = state.get('members', [])
    if (not members or [m['fold'] for m in members] != state['signature']['trainer']['folds'] or
            any(m.get('best') is None or not m.get('best_epoch', 0) or m.get('validation_pending', False) for m in members)):
        raise ValueError('Every ensemble member needs a completed validation and a selected checkpoint')
    models = []
    for member in members:
        model = GraphRegressor(encoder, graph)
        model.load_state_dict(member['best'], strict=True)
        models.append(model.cuda().eval())
    return models, graph, state


def infer(config, out):
    from grasppanda.compat import legacy_torch
    legacy_torch()
    import importlib.util
    import numpy as np
    import scipy.io
    import torch
    from torch_geometric.data import Batch
    from graspnetAPI import GraspNet, GraspGroup
    from grasppanda.methods.gtg2_data import candidates, native_geometry
    from grasppanda.modules.gtg2 import build_graph
    from grasppanda.worker import overlay
    models, graph, checkpoint = load_ensemble(config)
    g = GraspNet(config.dataset_root, camera=config.camera, split=config.split)
    crop = native_geometry()['to_gripper_coord']
    detector_class = None
    if config.collision_thresh:
        path = ROOT/catalogue()['graspnet_baseline']['path']/'utils/collision_detector.py'
        spec = importlib.util.spec_from_file_location('_grasppanda_gtg2_collision', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        detector_class = module.ModelFreeCollisionDetector
    records = []
    out.mkdir(parents=True, exist_ok=True)
    start_index = config.scene*256+config.frame
    for index in range(start_index, start_index+config.frames):
        scene, frame = divmod(index, 256)
        seed = (config.seed+index) % 2**32
        folder = Path(config.dataset_root)/'scenes'/f'scene_{scene:04d}'/config.camera
        paths = [folder/kind/f'{frame:04d}.{suffix}' for kind, suffix in
                 (('rgb', 'png'), ('depth', 'png'), ('label', 'png'), ('meta', 'mat'))]
        paths += [folder/'camera_poses.npy', folder/'cam0_wrt_table.npy']
        evidence = {str(p.relative_to(config.dataset_root)): digest(p) for p in paths}
        start = time.monotonic()
        rows, counts = candidates(g, config.dataset_root, scene, config.camera, frame, graph, seed)
        raw_cloud = g.loadScenePointCloud(sceneId=scene, camera=config.camera, annId=frame,
            use_workspace=True, align=False, use_inpainting=True)
        cloud = raw_cloud.voxel_down_sample(graph['voxel_size'])
        selected, scores, graphs, identifiers = [], [], [], []

        def score_batch():
            if not graphs: return
            batch = Batch.from_data_list(graphs).cuda()
            with torch.inference_mode():
                values = torch.stack([model(batch) for model in models]).mean(0)
            if values.shape != (len(graphs), 1) or not torch.isfinite(values).all():
                raise ValueError('Invalid graph ensemble predictions')
            selected.extend(identifiers); scores.extend(values[:, 0].cpu().tolist())
            graphs.clear(); identifiers.clear()

        for candidate, row in enumerate(rows):
            inside, outside, _, _ = crop(cloud, dict(t=row[13:16], R=row[4:13].reshape(3, 3),
                width=row[1], depth=row[3], score=0.), gripper_depth=graph['gripper_depth'],
                gripper_height=graph['gripper_height'], bound_size=graph['bound_size'])
            if len(inside) < graph['min_inside_infer']: continue
            graphs.append(build_graph(inside, outside, 0., graph))
            identifiers.append(candidate)
            if len(graphs) == config.batch_size: score_batch()
        score_batch()
        selected = np.asarray(selected, dtype=np.int64)
        predictions = rows[selected].copy()
        predictions[:, 0] = scores
        order = np.argsort(-predictions[:, 0], kind='stable')
        gg = GraspGroup(predictions[order])
        selected = selected[order]
        before = len(gg)
        if detector_class and len(gg):
            detector = detector_class(np.asarray(raw_cloud.points), voxel_size=config.voxel_size)
            keep = ~detector.detect(gg, approach_dist=.05, collision_thresh=config.collision_thresh)
            gg = gg[keep]; selected = selected[keep]
        destination = out/'predictions'/f'scene_{scene:04d}'/config.camera
        destination.mkdir(parents=True, exist_ok=True)
        path = destination/f'{frame:04d}.npy'
        gg.save_npy(str(path))
        if index == start_index:
            intr = scipy.io.loadmat(folder/'meta'/f'{frame:04d}.mat')['intrinsic_matrix']
            overlay(folder/'rgb'/f'{frame:04d}.png', gg.grasp_group_array, intr, out/'preview.png')
        if any(digest(Path(config.dataset_root)/p) != sha for p, sha in evidence.items()):
            raise ValueError('Frame inputs changed during candidate inference')
        row = dict(scene=scene, frame=frame, **counts, grasps_before_collision=before,
            grasps_after_collision=len(gg), pipeline_seconds=time.monotonic()-start,
            prediction=str(path.relative_to(out)), prediction_sha256=digest(path),
            input_sha256=evidence)
        if not len(gg): row['notice'] = 'No eligible grasps remain; inspect candidate geometry, checkpoint and collision settings.'
        records.append(row); print(json.dumps(row), flush=True)
    manifest = dict(config=config.to_dict(), checkpoint_sha256=digest(config.checkpoint),
        files={str(Path(r['prediction']).relative_to('predictions')): r['prediction_sha256'] for r in records})
    (out/'predictions/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return dict(stage='dataset_inference', method='gtg2', camera=config.camera, split=config.split,
        workspace=config.workspace, frames=records, checkpoint_sha256=digest(config.checkpoint),
        ensemble=[dict(fold=m['fold'], selected_epoch=m['best_epoch']) for m in checkpoint['members']],
        training_camera=checkpoint['signature']['camera'], ap=None,
        protocol_note='Reconstructed GtG2 candidate generation and trained mean-score ensemble; official GT workspace. Validation loss is not benchmark AP.',
        timing_note='Candidate generation, graph construction, ensemble scoring, filtering and output, including first-call overhead.')
