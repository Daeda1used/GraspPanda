"""Load the native FineGrasp model and its declared point operators in a worker."""
import importlib
import sys
import types


def native_module():
    from .config import ROOT, catalogue
    from .compat import legacy_torch
    root=ROOT/catalogue()['finegrasp']['path']
    if not (root/'robo_orchard_lab/version.py').is_file():
        raise ValueError('FineGrasp source metadata is missing; run the component installer')
    legacy_torch()
    sys.path.insert(0,str(root))
    ops=ROOT/'upstream/single_view/pointcloud/scale_balanced_grasp'
    import pointnet2
    pointnet2.__path__=list(pointnet2.__path__)+[str(ops/'pointnet2')]
    sys.path.insert(0,str(ops/'pointnet2'))
    # Author setup uses SBG's one-based KNN operator. The model still imports
    # its old package location; expose that exact wrapper at the expected path.
    package=types.ModuleType('robo_orchard_lab.ops.knn')
    package.__path__=[str(ops/'knn')]
    sys.modules[package.__name__]=package
    # Import the I/O processor without initializing unrelated training hooks
    # and robotics database integrations at processing package import time.
    if 'robo_orchard_lab.processing' not in sys.modules:
        package=types.ModuleType('robo_orchard_lab.processing')
        package.__path__=[str(root/'robo_orchard_lab/processing')]
        sys.modules[package.__name__]=package
    return importlib.import_module('robo_orchard_lab.models.finegrasp.finegrasp')


def infer(config, out):
    """Run native FineGrasp preprocessing, detector, decoding and collision/NMS."""
    import json
    import random
    import time
    from pathlib import Path
    import numpy as np
    import scipy.io
    import torch
    from safetensors.torch import load_file
    from .jobs import digest
    from .worker import overlay
    source=native_module()
    checkpoint=Path(config.checkpoint)
    metadata=checkpoint.parent/'model.config.json'
    settings=json.loads(metadata.read_text())
    allowed={'model_name','seed_feat_dim','graspness_threshold','grasp_max_width','num_depth','num_view',
             'num_angle','num_seed_points','voxel_size','cylinder_radius','cylinder_groups','use_normal','loss'}
    for name,expected in [('class_type','robo_orchard_lab.models.finegrasp.finegrasp:FineGrasp'),
                          ('__config_type__','robo_orchard_lab.models.finegrasp.finegrasp:FineGraspConfig')]:
        if settings.pop(name,expected)!=expected:raise ValueError('Unsupported FineGrasp model configuration type')
    if set(settings)-allowed or settings.get('loss') is not None:
        raise ValueError('Inference requires a FineGrasp architecture configuration with loss: null')
    model=source.FineGrasp(source.FineGraspConfig(**settings))
    model.load_state_dict(load_file(str(checkpoint)),strict=True)
    model.cuda().eval()
    from robo_orchard_lab.models.finegrasp.processor import GraspInput,FineGraspProcessor,FineGraspProcessorCfg
    processor=FineGraspProcessor(FineGraspProcessorCfg(voxel_size=config.voxel_size,
        grasp_max_width=model.cfg.grasp_max_width,num_seed_points=model.cfg.num_seed_points,
        max_gripper_width=model.cfg.grasp_max_width,collision_thresh=config.collision_thresh))
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    rows=[]
    bounds=[-1.,1.,-1.,1.,0.,2.]
    for index in range(config.scene*256+config.frame,config.scene*256+config.frame+config.frames):
        scene,frame=divmod(index,256)
        folder=Path(config.dataset_root)/'scenes'/f'scene_{scene:04d}'/config.camera
        rgb=folder/'rgb'/f'{frame:04d}.png';depth=folder/'depth'/f'{frame:04d}.png';meta=folder/'meta'/f'{frame:04d}.mat'
        calibration=scipy.io.loadmat(meta)
        intr=calibration['intrinsic_matrix']
        inputs=GraspInput(rgb_image=str(rgb),depth_image=str(depth),intrinsic_matrix=intr,
                         depth_scale=float(np.asarray(calibration.get('factor_depth',1000)).item()),
                         grasp_workspace=bounds,num_sample_points=config.num_points)
        start=time.monotonic()
        data=processor.pre_process(inputs,device='cuda')
        with torch.no_grad():prediction=model(data)
        grasps=processor.post_process(prediction,data).grasp_poses
        array=grasps.grasp_group_array
        if array.ndim!=2 or array.shape[1]!=17 or not np.isfinite(array).all():raise ValueError('Invalid native FineGrasp output')
        target=out/'predictions'/f'scene_{scene:04d}'/config.camera/f'{frame:04d}.npy'
        target.parent.mkdir(parents=True,exist_ok=True);grasps.save_npy(str(target))
        torch.cuda.synchronize()
        rows.append(dict(scene=scene,frame=frame,grasps_saved=len(grasps),sampled_points=config.num_points,
            pipeline_seconds=time.monotonic()-start,prediction=str(target.relative_to(out)),prediction_sha256=digest(target),
            rgb_sha256=digest(rgb),depth_sha256=digest(depth),meta_sha256=digest(meta)))
        if len(rows)==1:overlay(rgb,array,intr,out/'preview.png')
    manifest=dict(config=config.to_dict(),checkpoint_sha256=digest(checkpoint),model_config_sha256=digest(metadata),
        files={str(Path(r['prediction']).relative_to('predictions')):r['prediction_sha256'] for r in rows})
    (out/'predictions/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(stage='dataset_inference',method=config.method,camera=config.camera,split=config.split,workspace=config.workspace,
        frames=rows,checkpoint_sha256=manifest['checkpoint_sha256'],model_config_sha256=manifest['model_config_sha256'],
        workspace_bounds=bounds,ap=None,protocol_note='Native RGB-D and estimated normals, camera calibration, fixed camera-space bounds in metres, native collision filtering and NMS; no GT segmentation.')
