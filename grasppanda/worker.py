"""One experiment per process, using the same shared Python environment."""
import importlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time

from .compat import legacy_torch
from .config import ROOT, Experiment, catalogue, probes
from .jobs import digest


def prepare(method):
    repo = ROOT / catalogue()[method]["path"]
    os.chdir(repo)
    sys.path[:0] = [str(repo / p) for p in ("", "models", "dataset", "utils", "pointnet2", "knn")]
    legacy_torch()
    return repo


def frame_cloud(config, scene, frame):
    from .datasets import get_provider
    return get_provider(config.dataset).frame_cloud(config,scene,frame)


def overlay(rgb_path, grasps, intr, destination):
    """Project gripper axes onto RGB for inspection; this is not collision validation."""
    import numpy as np
    from PIL import Image, ImageDraw
    if not rgb_path.exists():
        return
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for grasp in sorted(grasps, key=lambda g: -g[0])[:30]:
        rotation = grasp[4:13].reshape(3, 3)
        center, width, depth = grasp[13:16], grasp[1], grasp[3]
        # GraspNet gripper convention: approach x, opening y, height z.
        local = np.array([[depth, -width/2, 0], [-0.02, -width/2, 0], [-0.02, width/2, 0], [depth, width/2, 0]])
        xyz = local @ rotation.T + center
        if (xyz[:, 2] <= 0).any():
            continue
        uvw = xyz @ intr.T
        uv = uvw[:, :2] / uvw[:, 2:]
        draw.line([tuple(p) for p in uv], fill=(61, 225, 161), width=2)
    image.save(destination)


def infer(config, out):
    if config.method=='finegrasp':
        from .finegrasp import infer as finegrasp_infer
        return finegrasp_infer(config,out)
    if config.method in ('economicgrasp', 'dograspnet'):
        from .native_points import infer as native_infer
        return native_infer(config, out)
    if config.method in ('hggd', 'region_normalized_grasp'):
        from .heatmap import infer as heatmap_infer
        return heatmap_infer(config, out)
    import numpy as np
    import torch
    from graspnetAPI import GraspGroup
    repo = prepare(config.method)
    if config.method == 'graspbalance':
        sys.path[:0] = [str(repo/p) for p in ('TrainModel','PointNet','KNN','DataProcessing','ModifiedNetTools')]
    if config.method == 'graspfast':
        from .graspfast import prepare as prepare_graspfast
        module = prepare_graspfast()
        model, pred_decode = module.GraspFast(is_training=False), module.pred_decode
    elif config.method == 'graspbalance':
        module = importlib.import_module('TrainModel.graspbalance')
        model, pred_decode = module.GraspBalance(is_training=False), module.pred_decode
    elif config.method == 'granet':
        from .compat import legacy_dgl
        legacy_dgl()
        module = importlib.import_module('models.granet_pipeline')
        model, pred_decode = module.GraNet(batch_size=1,is_training=False), module.pred_decode
    elif config.method == "fgc_graspnet":
        from models.FGC_graspnet import FGC_graspnet
        from models.decode import pred_decode
        model = FGC_graspnet(is_training=False, is_demo=True)
    else:
        module = importlib.import_module("models.graspnet")
        cls = module.GraspNet_MSCQ if config.method == "scale_balanced_grasp" else module.GraspNet
        model, pred_decode = cls(is_training=False, **({'backbone':'resunet'} if config.method=='graspness_modern' else {})), module.pred_decode
    state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    if config.method == 'graspfast':
        from .graspfast import checkpoint_state
        state = checkpoint_state(state)
    if config.method == 'graspness_modern':
        state = {k.removeprefix('module.'):v for k,v in state.get('model_state_dict',state).items()}
        state.setdefault('rotation.template_views',model.rotation.template_views.detach().cpu())
    from .components import configure_model,load_checkpoint
    prefixes=configure_model(model,config.method,config.modules,config.voxel_size)
    transfer=load_checkpoint(model,state.get("model_state_dict",state),prefixes,config.checkpoint_policy)
    (out/'component_transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
    model.cuda().eval()
    if config.method == 'graspfast':
        from .graspfast import guard as guard_graspfast
        guard_graspfast(model, module)
    if config.method in ('graspness','graspness_modern'):
        def guard(_module, _inputs, end):
            counts = ((end["objectness_score"].argmax(1) == 1) & (end["graspness_score"].squeeze(1) > module.GRASPNESS_THRESHOLD)).sum(1)
            if (counts == 0).any():
                raise ValueError("No graspable points: refusing unsafe upstream FPS on an empty set")
        model.graspable.register_forward_hook(guard)
    from collision_detector import ModelFreeCollisionDetector
    from .datasets import get_dataset
    frame_count=get_dataset(config.dataset).frames_per_scene
    records = []
    for index in range(config.scene * frame_count + config.frame, config.scene * frame_count + config.frame + config.frames):
        scene, frame = divmod(index, frame_count)
        raw, xyz, intr, rgb, evidence = frame_cloud(config, scene, frame)
        inputs = {"point_clouds": torch.from_numpy(xyz)[None].cuda()}
        if config.method in ('graspness', 'graspfast'):
            import MinkowskiEngine as ME
            coords, feats = ME.utils.sparse_collate([xyz / config.voxel_size], [np.ones_like(xyz)])
            coords, feats, _, inverse = ME.utils.sparse_quantize(coords, feats, return_index=True, return_inverse=True)
            inputs.update(coors=coords.cuda(), feats=feats.cuda(), quantize2original=inverse.cuda())
        if config.method == 'graspness_modern':
            from dataset.graspnet_dataset import spconv_collate_fn
            inputs = {k:v.cuda() for k,v in spconv_collate_fn([dict(point_clouds=xyz,
                coors=xyz/config.voxel_size,feats=np.ones_like(xyz))]).items()}
        if config.method == 'granet':
            from dataset.graph_generator import GraphGenerator
            inputs['graph'] = [GraphGenerator().init_knn_graph({'point_clouds':xyz})['graph'].to('cuda')]
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            predictions = pred_decode(model(inputs))[0].detach().cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if predictions.ndim != 2 or predictions.shape[1] != 17 or not np.isfinite(predictions).all():
            raise ValueError("Invalid GraspNet prediction tensor")
        gg = GraspGroup(predictions)
        raw_count = len(gg)
        if config.collision_thresh > 0 and len(gg):
            detector = ModelFreeCollisionDetector(raw, voxel_size=0.01)
            gg = gg[~detector.detect(gg, approach_dist=0.05, collision_thresh=config.collision_thresh)]
        destination = out / "predictions" / f"scene_{scene:04d}" / config.camera
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{frame:04d}.npy"
        gg.save_npy(str(path))
        if index == config.scene * frame_count + config.frame:
            overlay(rgb, gg.grasp_group_array, intr, out / "preview.png")
        record = dict(scene=scene, frame=frame, raw_points=len(raw), sampled_points=len(xyz),
                      grasps_before_collision=raw_count, grasps_after_collision=len(gg),
                      forward_decode_seconds=elapsed, prediction=str(path.relative_to(out)),
                      prediction_sha256=digest(path), **evidence)
        if not raw_count:
            record['notice']='The native decoder returned no candidates. Newly initialized components may need substantially more training; inspect input units and checkpoint/composition before judging quality. This is not zero benchmark AP.'
        elif not len(gg):
            record['notice']='Collision filtering removed all candidates. Inspect the preview, input geometry and collision settings; this is not zero benchmark AP.'
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest = {'config': config.to_dict(), 'checkpoint_sha256':digest(config.checkpoint),
                'files':{str(Path(r['prediction']).relative_to('predictions')):r['prediction_sha256'] for r in records}}
    (out/'predictions/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {"stage": "dataset_inference", "method": config.method, "camera": config.camera,
            "split": config.split, "workspace": config.workspace, "frames": records,
            "checkpoint_sha256": digest(config.checkpoint), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(), "ap": None,
            "timing_note": "Forward + decode including first-call overhead; not a warmed throughput benchmark",
            "protocol_note": "official_gt_workspace uses dataset segmentation and poses for workspace cropping; depth_only is a separate protocol"}


def train(config, out):
    if config.method=='finegrasp':
        from .training_finegrasp import run
        return run(config,out)
    from .native_training import run
    return run(config,out)


def train_smoke(config, out):
    """One real-label optimizer step, using upstream data/model/loss code."""
    if config.method=='finegrasp':
        from .training_finegrasp import run
        return run(config,out,1)
    import numpy as np
    import torch
    import scipy.io
    prepare(config.method)
    dataset_module = importlib.import_module('graspnet_dataset')
    model_module = importlib.import_module('models.graspnet')
    scene = f'scene_{config.scene:04d}'
    root = Path(config.dataset_root)
    directory = root/'scenes'/scene/config.camera
    meta = scipy.io.loadmat(directory/'meta'/f'{config.frame:04d}.mat')
    ids = meta['cls_indexes'].flatten().astype(int).tolist()
    labels = {}
    evidence = {}
    for obj in ids:
        if obj == 19 and config.method != 'graspness':
            continue  # Same invalid-object exclusion as the baseline loader.
        name = f'{obj-1:03d}'
        labelpath = root/('grasp_label_simplified' if config.method == 'graspness' else 'grasp_label')/(name+'_labels.npz')
        with np.load(labelpath) as label:
            if config.method == 'graspness':
                labels[obj] = tuple(label[k].astype(np.float32) for k in ('points','width','scores'))
            else:
                tolerance_path = root/'tolerance'/(name+'_tolerance.npy')
                labels[obj] = tuple(label[k].astype(np.float32) for k in ('points','offsets','scores')) + (np.load(tolerance_path),)
                evidence['tolerance_'+name] = digest(tolerance_path)
        evidence['labels_'+name] = digest(labelpath)
    kwargs = dict(root=config.dataset_root, grasp_labels=labels, camera=config.camera, split='train',
                  num_points=config.num_points, remove_outlier=True, augment=False, load_label=False)
    if config.method == 'graspness':
        kwargs['voxel_size'] = config.voxel_size
    else:
        kwargs['valid_obj_idxs'] = list(labels)
    dataset = dataset_module.GraspNetDataset(**kwargs)
    collision_path = root/'collision_label'/scene/'collision_labels.npz'
    with np.load(collision_path) as collision:
        dataset.collision_labels[scene] = {i: collision[f'arr_{i}'] for i in range(len(collision))}
    evidence['collision'] = digest(collision_path)
    dataset.load_label = True
    data = dataset[config.scene*256+config.frame]
    if not data['object_poses_list']:
        raise ValueError('No labelled object survived sampling; increase num_points')
    collate = dataset_module.minkowski_collate_fn if config.method == 'graspness' else dataset_module.collate_fn
    batch = collate([data])
    def cuda(value):
        if isinstance(value, torch.Tensor): return value.cuda()
        if isinstance(value, dict): return {k:cuda(v) for k,v in value.items()}
        if isinstance(value, list): return [cuda(v) for v in value]
        return value
    batch = cuda(batch)
    cls = model_module.GraspNet_MSCQ if config.method == 'scale_balanced_grasp' else model_module.GraspNet
    model = cls(is_training=True).cuda().train()
    from .components import configure_model,load_checkpoint
    prefixes=configure_model(model,config.method,config.modules,config.voxel_size)
    model.cuda()
    if config.checkpoint:
        state = torch.load(config.checkpoint, map_location='cpu', weights_only=True)
        transfer=load_checkpoint(model,state.get('model_state_dict',state),prefixes,config.checkpoint_policy)
        (out/'component_transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
    if config.method == 'graspness':
        def guard(_module,_inputs,end):
            counts=((end['objectness_score'].argmax(1)==1)&(end['graspness_score'].squeeze(1)>model_module.GRASPNESS_THRESHOLD)).sum(1)
            if (counts==0).any(): raise ValueError('No graspable points; refusing empty upstream FPS')
        model.graspable.register_forward_hook(guard)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    print('Running upstream forward, loss and backward on one labelled frame', flush=True)
    output = model(batch)
    loss, _ = importlib.import_module('models.loss').get_loss(output)
    if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing or non-finite gradients')
    optimizer.step()
    torch.cuda.synchronize()
    return {'stage':'dataset_training_step', 'method':config.method, 'scene':config.scene, 'frame':config.frame,
            'loss':loss.item(), 'parameters_with_gradient':len(gradients), 'optimizer_steps':1,
            'label_sha256':evidence, 'note':'Single frame/batch=1; no augmentation. Not convergence or full trainer validation.'}


def evaluate(config, out):
    from .datasets import get_provider
    return get_provider(config.dataset).evaluate(config,out)


def main():
    config_path, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    config = Experiment.from_dict(json.loads(config_path.read_text())).preflight()
    provenance_path = out/'provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance['config_sha256'] != digest(config_path):
            raise ValueError('Queued configuration changed before execution')
        if provenance['runtime_lock_sha256'] != digest(ROOT/'uv.lock'):
            raise ValueError('Runtime lock changed while job was queued; resubmit against the new runtime')
        if provenance.get('native_source_lock_sha256') and provenance['native_source_lock_sha256']!=digest(ROOT/'grasppanda/resources/native_sources.lock.json'):
            raise ValueError('Native source lock changed while queued; resubmit')
        for path,expected in provenance.get('native_sources',{}).items():
            actual=subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()
            if actual!=expected:raise ValueError('Native dependency revision changed while queued: '+path)
        if 'component_source_lock_sha256' in provenance:
            component_lock=ROOT/'grasppanda/resources/component_sources.lock.json'
            if digest(component_lock)!=provenance['component_source_lock_sha256']:
                raise ValueError('Component source lock changed while queued')
            components={r['id']:r for r in json.loads(component_lock.read_text())}
            for name,expected in provenance.get('component_sources',{}).items():
                path=ROOT/components[name]['path']
                actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
                if actual!=expected:raise ValueError('Component source revision changed while queued: '+name)
        if provenance.get('checkpoint_sha256') and digest(config.checkpoint) != provenance['checkpoint_sha256']:
            raise ValueError('Queued checkpoint changed before execution')
        if provenance.get('model_config_sha256') and digest(Path(config.checkpoint).parent/'model.config.json')!=provenance['model_config_sha256']:
            raise ValueError('Queued model configuration changed before execution')
        for path,expected in provenance.get('recipe_weights',{}).items():
            if digest(ROOT/path)!=expected: raise ValueError('Recipe weight changed while queued; resubmit')
        for path,expected in provenance.get('toolbox_sources', provenance.get('workbench_sources', {})).items():
            if digest(ROOT/path)!=expected: raise ValueError('Toolbox code changed while queued; resubmit')
        repo=ROOT/catalogue()[config.method]['path']
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if commit!=provenance['upstream_commit']: raise ValueError('Upstream revision changed while queued')
    import numpy as np
    import torch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(4)
    if config.action == "probe":
        module, forward = probes()[config.method]
        command = [sys.executable, str(ROOT / "grasppanda/runtime/probe_model.py"), config.method, "--module", module]
        if forward:
            command.append("--forward")
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        marker = next(line for line in completed.stdout.splitlines() if line.startswith("GRASPPANDA_RESULT="))
        result = json.loads(marker.split("=", 1)[1])
    elif config.action == 'train_check':
        from .training import hggd,point_family,rng,contact,rgb_matters
        from .training_center import run as center
        from .training_gfla import run as gfla
        from .training_motion import run as motion
        from .training_spahybgen import run as spahybgen
        from .training_graspfast import run as graspfast
        from .training_finegrasp import run as finegrasp
        runner={'finegrasp':finegrasp,'hggd':hggd,'region_normalized_grasp':rng,'contact_graspnet_g1b':contact,'rgb_matters':rgb_matters,'centergrasp':center,'gfla':gfla,'motiongrasp':motion,'spahybgen':spahybgen,'graspfast':graspfast}.get(config.method,point_family)
        result=runner(config,out,config.training_steps)
    elif config.action == 'pipeline_smoke':
        command=[sys.executable,str(ROOT/'grasppanda/runtime/run_recipe.py'),config.method,'--dataset-root',config.dataset_root,'--out',str(out)]
        if config.checkpoint:command+=['--checkpoint',config.checkpoint]
        subprocess.run(command,cwd=ROOT,check=True)
        result=json.loads((out/'result.json').read_text())
    else:
        result = {"infer": infer, "train": train, "train_smoke": train_smoke, "evaluate": evaluate}[config.action](config, out)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GRASPPANDA_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
