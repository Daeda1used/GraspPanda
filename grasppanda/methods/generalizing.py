"""Visual fused-scene MSCQ inference with the author's normals and decoder."""
from pathlib import Path
import json
import time


def source_path(config):
    folder=Path(config.dataset_root)/'fusion_scenes'/f'scene_{config.scene:04d}'/config.camera
    numeric=folder/'points.npz'
    return numeric if numeric.is_file() else folder/'points.npy'


def cloud(config):
    import numpy as np
    from .fusion_data import read_fusion
    data=read_fusion(source_path(config))
    if config.camera=='kinect':
        xyz=data['xyz'];mask=((xyz[:,0]>-.5)&(xyz[:,0]<.5)&(xyz[:,1]>-.5)&(xyz[:,1]<.5)&(xyz[:,2]>-.02)&(xyz[:,2]<.2))
        data={key:value[mask] for key,value in data.items()}
    if not len(data['xyz']):raise ValueError('The fused scene contains no points in the native workspace')
    return data


def preflight(config):
    if not source_path(config).is_file():raise ValueError('Generalizing-Grasp needs fusion_scenes/scene_XXXX/CAMERA/points.npy or points.npz; see Data & weights')
    if not config.checkpoint or not Path(config.checkpoint).is_file():raise ValueError('Select a matching Generalizing-Grasp checkpoint')


def infer(config, out):
    import numpy as np
    import torch
    from types import FunctionType,MethodType
    from ..worker import prepare,overlay
    from ..jobs import digest
    prepare(config.method)
    from mink_dataset import minkowski_collate_fn
    from graspnet_sparseconv import GraspNet_MSCQ,pred_decode
    from collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup
    model=GraspNet_MSCQ(is_training=False).cuda().eval()
    state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model_state_dict'],strict=True)
    stage=model.view_estimator.vpmodule
    original=stage.forward.__func__;namespace=dict(original.__globals__)
    fps=namespace['furthest_point_sample']
    def safe_fps(points,count):
        if points.shape[1]==0:raise ValueError('The fused model predicted no graspable points; check its input and checkpoint')
        return fps(points,count)
    namespace['furthest_point_sample']=safe_fps
    stage.forward=MethodType(FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__),stage)
    data=cloud(config);raw=data['xyz'].astype(np.float32)
    if len(raw)>=config.num_points:ids=np.random.choice(len(raw),config.num_points,replace=False)
    else:ids=np.concatenate((np.arange(len(raw)),np.random.choice(len(raw),config.num_points-len(raw),replace=True)))
    xyz=raw[ids]
    sample=dict(point_clouds=xyz,coors=xyz/config.voxel_size,feats=data['normal'][ids].astype(np.float32),
                pcd_color=np.concatenate((xyz,data['color'][ids].astype(np.float32)),axis=1))
    batch={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in minkowski_collate_fn([sample]).items()}
    torch.cuda.synchronize();start=time.monotonic()
    with torch.inference_mode():grasps=pred_decode(model(batch))[0][0].cpu().numpy()
    torch.cuda.synchronize();seconds=time.monotonic()-start
    if grasps.ndim!=2 or grasps.shape[1]!=17 or not np.isfinite(grasps).all():raise ValueError('Invalid fused-scene predictions')
    before=len(grasps)
    if config.collision_thresh>0 and len(grasps):
        detector=ModelFreeCollisionDetector(raw,voxel_size=.01)
        grasps=grasps[~detector.detect(GraspGroup(grasps),approach_dist=.05,collision_thresh=config.collision_thresh)]
    folder=out/'predictions'/f'scene_{config.scene:04d}'/config.camera;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'result.npy';np.save(path,grasps)
    row=dict(scene=config.scene,frame=0,raw_points=len(raw),sampled_points=len(xyz),
             grasps_before_collision=before,grasps_after_collision=len(grasps),forward_decode_seconds=seconds,
             prediction=str(path.relative_to(out)),prediction_sha256=digest(path),fusion_points_sha256=digest(source_path(config)))
    camera=Path(config.dataset_root)/'scenes'/f'scene_{config.scene:04d}'/config.camera
    if all((camera/name).exists() for name in ('meta/0000.mat','rgb/0000.png','cam0_wrt_table.npy')):
        import scipy.io
        from ..refinement.pipeline import transform_grasps
        table=np.load(camera/'cam0_wrt_table.npy',allow_pickle=False)
        intr=scipy.io.loadmat(camera/'meta/0000.mat')['intrinsic_matrix']
        overlay(camera/'rgb/0000.png',transform_grasps(grasps,np.linalg.inv(table)),intr,out/'preview.png')
    manifest=dict(config=config.to_dict(),checkpoint_sha256=digest(config.checkpoint),coordinate_frame='table',
                  observation='multi_view_fusion',files={str(path.relative_to(out/'predictions')):digest(path)})
    (out/'predictions/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(stage='fused_scene_inference',method=config.method,camera=config.camera,split=config.split,
                workspace=config.workspace,coordinate_frame='table',observation='multi_view_fusion',frames=[row],
                checkpoint_sha256=manifest['checkpoint_sha256'],ap=None,
                protocol_note='Native fused XYZ/normals and MSCQ decoder; no GT instance masks. One fused scene per job; scene sweeps cover larger ranges.')
