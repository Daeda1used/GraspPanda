"""Bounded, isolated reproduction attempts; every result states its actual scope.

Run through run_attempts.py to retain failures, timeouts and source provenance.
This is a diagnostic runner, not an AP benchmark.
"""
import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasppanda.config import catalogue
from grasppanda.compat import legacy_torch
from grasppanda.jobs import digest


def definitions(path, names, namespace):
    """Load verbatim upstream definitions without unrelated CLI/module side effects."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names): raise ValueError(f'Missing upstream definitions: {names}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        from grasppanda.recipes import CHECKPOINT_RECIPES
        if args.method not in CHECKPOINT_RECIPES:raise ValueError('Checkpoint override is not integrated for this recipe')
        args.checkpoint=args.checkpoint.resolve()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    mid = args.method; repo = ROOT/catalogue()[mid]['path']; os.chdir(repo)
    sys.path[:0] = [str(repo/p) for p in ('', 'models', 'dataset', 'utils', 'pointnet2', 'knn')]
    sys.argv = ['diagnostic.py']; legacy_torch()
    import torch
    import numpy as np
    import open3d as o3d
    import scipy.io
    from PIL import Image
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    import random
    random.seed(0)
    camera = 'realsense'
    directory = Path(args.dataset_root)/'scenes/scene_0100'/camera
    result = dict(method=mid, ap=None, checkpoint=None)
    if args.checkpoint:
        result.update(checkpoint='experiment-selected checkpoint',checkpoint_sha256=digest(args.checkpoint))
    start = time.monotonic()
    def checkpoint(name='checkpoint-realsense.tar'):
        p = args.checkpoint or ROOT/'checkpoints'/mid/name
        result['checkpoint'] = 'experiment-selected checkpoint' if args.checkpoint else str(p.relative_to(ROOT)); result['checkpoint_sha256'] = digest(p)
        return torch.load(p, map_location='cpu', weights_only=True)
    def real_xyz(n=4096):
        depth = np.array(Image.open(directory/'depth/0000.png'))/1000.
        k = scipy.io.loadmat(directory/'meta/0000.mat')['intrinsic_matrix']
        yy, xx = np.indices(depth.shape)
        cloud = np.stack([(xx-k[0,2])*depth/k[0,0], (yy-k[1,2])*depth/k[1,1],depth],-1)
        valid = depth>0; xyz = cloud[valid].astype(np.float32)
        return xyz[np.random.choice(len(xyz),n,replace=False)]
    def save_grasps(array):
        if isinstance(array, torch.Tensor): array = array.detach().cpu().numpy()
        assert array.ndim == 2 and array.shape[1] == 17 and np.isfinite(array).all()
        np.save(out/'grasps.npy', array); result.update(grasps=len(array),prediction_sha256=digest(out/'grasps.npy'))

    if mid == 'graspbalance':
        sys.path[:0] = [str(repo/p) for p in ('TrainModel','PointNet','KNN','DataProcessing','ModifiedNetTools')]
        mod = importlib.import_module('TrainModel.graspbalance')
        model = mod.GraspBalance(is_training=False).cuda().eval()
        with torch.no_grad(): pred = mod.pred_decode(model({'point_clouds':torch.from_numpy(real_xyz(15000))[None].cuda()}))
        save_grasps(pred[0]); result['stage']='real_input_random_weight_forward_decode'
    elif mid == 'granet':
        from dataset.graph_generator import GraphGenerator
        from models.granet_pipeline import GraNet, pred_decode
        xyz = real_xyz(12000)
        graph = GraphGenerator().init_knn_graph({'point_clouds':xyz})['graph'].to('cuda')
        model = GraNet(batch_size=1,is_training=False).cuda().eval()
        with torch.no_grad(): pred = pred_decode(model({'point_clouds':torch.from_numpy(xyz)[None].cuda(),'graph':[graph]}))
        save_grasps(pred[0]); result['stage']='real_input_random_weight_forward_decode'
    elif mid == 'generalizing_grasp':
        from mink_dataset import GraspNetDataset_fusion, minkowski_collate_fn
        from graspnet_sparseconv import GraspNet_MSCQ, pred_decode
        dataset = GraspNetDataset_fusion(args.dataset_root, valid_obj_idxs=None, grasp_labels=None,
            split='test', camera=camera, num_points=20000, remove_outlier=True, augment=False, load_label=False,use_fine=False)
        model = GraspNet_MSCQ(is_training=False).cuda().eval()
        model.load_state_dict(checkpoint('model.tar')['model_state_dict'],strict=True)
        sample = dataset[0]; batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in minkowski_collate_fn([sample]).items()}
        with torch.no_grad(): pred, saved = pred_decode(model(batch))
        save_grasps(pred[0]); result.update(stage='fused_scene_forward_decode',protocol='upstream fused scene, GT segment labels, no C-SJO or AP')
    elif mid == 'contact_graspnet_g1b':
        sys.path.remove(str(repo/'utils'))
        from models.cgnet import ContactGraspNet
        from graspnetAPI import GraspNet, GraspGroup
        model=ContactGraspNet(SimpleNamespace()).cuda().eval()
        model.load_state_dict(checkpoint()['base_model'],strict=True)
        ns = dict(np=np,torch=torch,os=os,o3d=o3d,GraspNet=GraspNet,GraspGroup=GraspGroup)
        definitions(repo/'data/g1b.py',['GraspNet1BLoader'],ns)
        definitions(repo/'test.py',['inference_single'],ns)
        loader=ns['GraspNet1BLoader'](root=args.dataset_root,camera=camera,split='test_seen')
        xyz,_ = loader.loadScenePointCloud(100,camera,0,format='numpy')
        a=SimpleNamespace(device='cuda',visualize=False,dump_dir=str(out),camera=camera)
        ns['inference_single'](model,xyz,a,None,'scene_0100','0000')
        gg=GraspGroup(np.load(out/'scene_0100/realsense/0000.npy')); before=len(gg)
        from utils.collision_detector import ModelFreeCollisionDetectorGPU
        mask=ModelFreeCollisionDetectorGPU(torch.from_numpy(xyz.astype(np.float32)).cuda(),voxel_size=.01).detect(gg,approach_dist=.05,collision_thresh=.01)
        save_grasps(gg[~mask].grasp_group_array)
        result.update(stage='dataset_inference',grasps_before_collision=before,protocol='native camera-coordinate GT bounding box, centering, 20000 points, native collision')
    elif mid == 'rngnet_sdk':
        import RNGNet
        model=RNGNet.RngNet(checkpoint_path=str(repo/'realsense.pth'),camera=camera,use_anchornet=True)
        rgb=np.array(Image.open(directory/'rgb/0000.png'));depth=np.array(Image.open(directory/'depth/0000.png'))
        pred=model.infer_from_rgbd_heatmap(rgb,depth)
        array=np.array([[g.score,g.width,g.height,g.depth,*g.rotation.reshape(-1),*g.translation,g.object_id] for g in pred]).reshape(-1,17)
        save_grasps(array);result.update(stage='native_sdk_inference',protocol='Bundled RealSense RGB-D SDK weights, original grouping/decoding/postprocessing')
    elif mid == 'zerograsp':
        # Pinned OFE added scene-object ranges; this demo contains one scene.
        # Kernel ranges address masks, not point indices. All object masks belong
        # to this one image, so every range is [0, number_of_masks).
        from ofe import OctreeFeatureExtractor
        ofe_forward=OctreeFeatureExtractor.forward
        def single_scene_ofe(self, pts, masks, depth, intr, batch_id, grid_size):
            count=masks.shape[0]
            starts=torch.zeros(count,dtype=torch.int32,device=pts.device)
            ends=torch.full((count,),count,dtype=torch.int32,device=pts.device)
            return ofe_forward(self,pts,masks,depth,intr,batch_id,starts,ends,grid_size)
        OctreeFeatureExtractor.forward=single_scene_ofe
        # Original headless demo writes relative to demo/, redirected to this run.
        sys.argv=['demo.py','--img_path',str(repo/'demo/000.rgb.png'),'--depth_path',str(repo/'demo/000.depth.png'),
                  '--mask_path',str(repo/'demo/000.mask.png'),'--camera_info_path',str(repo/'demo/000.meta.yml'),
                  '--checkpoint',str(ROOT/'checkpoints/zerograsp/checkpoint-realsense.tar'),'--config',str(repo/'configs/demo.yaml')]
        os.chdir(out); (out/'demo').mkdir(exist_ok=True)
        runpy.run_path(str(repo/'demo.py'),run_name='__main__')
        predictions=list((out/'demo').glob('*.grasp.npy'))
        if not predictions: raise ValueError('Demo returned without any grasp artifact')
        result.update(stage='author_sample_full_inference',files=[p.name for p in predictions],protocol='synthetic-trained checkpoint; supplied RGB-D and instance mask; not G1B AP')
    elif mid == 'gfla':
        from grasppanda.overlays import prepare_overlay
        sys.path.insert(0,str(prepare_overlay('gfla')));sys.path.insert(0,str(ROOT/'environments/extensions'))
        sys.path.insert(0,str(prepare_overlay('gfla_source')))
        os.environ['GRASPPANDA_WEIGHTS_ROOT']=str(ROOT/'checkpoints/gfla')
        import grasp_nms_cpp
        sys.modules['models.my_grasp_nms.grasp_nms_cpp']=grasp_nms_cpp
        from LauncherTemplate import load_yaml
        import models.CFG_main as mod
        cfg=load_yaml('cfg/config.yaml')
        model=mod.CFGNet(cfg).cuda().eval();state=checkpoint()
        expected=set(model.state_dict())
        result['unused_checkpoint_keys']=sorted(set(state)-expected)
        model.load_state_dict({k:v for k,v in state.items() if k in expected},strict=True)
        rgb=np.array(Image.open(directory/'rgb/0000.png'))
        depth=torch.from_numpy(np.array(Image.open(directory/'depth/0000.png')).astype(np.float32))[None,None].cuda()/1000.
        intr=torch.from_numpy(scipy.io.loadmat(directory/'meta/0000.mat')['intrinsic_matrix'].astype(np.float32))[None].cuda()
        trans=torch.from_numpy((np.load(directory/'cam0_wrt_table.npy')@np.load(directory/'camera_poses.npy')[0]).astype(np.float32))[None].cuda()
        with torch.no_grad(): pred=model([rgb],depth,intr,trans)
        # Reuse the author's entire analytic decoder with the already computed
        # network output. Training-only dataset targets are not needed here.
        mod._run_model=lambda *_:(pred,{'Ts_Cn_in_T':trans})
        predictor=SimpleNamespace(model=model,config=load_yaml('cfg/trainer_config.yaml'),_visualize_grasp_in_scene=False,plot_mode=False)
        with torch.no_grad():decoded=mod.CFG_Predictor.run_model(predictor,None)
        save_grasps(decoded[0]);result.update(stage='dataset_inference',protocol='native RGB-D perception, SDF, analytic force closure, collision and NMS; author table transform')
    elif mid == 'active_ngf':
        import yaml
        cfg=yaml.safe_load((repo/'configs/GraspNet/scene_0100.yaml').read_text())
        cfg['data']['input_folder']=str(directory);cfg['data']['output']=str(out)
        cfg['grasp_checkpoint']=str(ROOT/'checkpoints/active_ngf/checkpoint-realsense.tar')
        cfg['mapping'].update(iters_first=150,iters=50,no_mesh_on_first_frame=True)
        p=out/'config.yaml';p.write_text(yaml.safe_dump(cfg))
        os.environ['WANDB_MODE']='disabled'
        from src import config as active_config
        from src.ESLAM import ESLAM
        import wandb
        full_cfg=active_config.load_config(str(p),'configs/ESLAM.yaml')
        engine=ESLAM(full_cfg,SimpleNamespace(input_folder=str(directory),output=str(out)),wandb.init(mode='disabled'))
        engine.mapper.max_step=1
        engine.run()
        result.update(stage='two_frame_active_mapping',protocol='Original 150/50 optimization iterations, mapping + next-view selection + meshing; stop after 2 mapped frames; no AP')
    elif mid == 'centergrasp':
        import mplib
        sys.modules['mplib.pymp.fcl']=mplib.pymp.collision_detection.fcl
        from grasppanda.overlays import prepare_overlay
        sys.path.insert(0,str(prepare_overlay('centergrasp')))
        import centergrasp.data_utils as du
        du.get_checkpoint_path = lambda name, folder: (args.checkpoint if args.checkpoint and folder=='ckpt_rgb' and name=='el6oa23g' else next((ROOT/'checkpoints/centergrasp'/folder/name).glob('*.ckpt')))
        import centergrasp.rgb.training_centergrasp as train_rgb
        load_config=train_rgb.load_rgb_config
        def graspnet_config():
            specs,cfg=load_config();specs['EmbeddingCkptPath']='6953cfxt';return specs,cfg
        train_rgb.load_rgb_config=graspnet_config
        from centergrasp.rgb.rgb_inference import RGBInference
        model=RGBInference('el6oa23g')
        import cv2
        directory=directory.parent/'kinect'
        rgb=np.ascontiguousarray(np.array(Image.open(directory/'rgb/0000.png'))[::2,::2][4:-4])
        depth=np.array(Image.open(directory/'depth/0000.png')).astype(np.float32)[::2,::2][4:-4,...,None]/1000.
        pred=model.get_full_predictions(rgb,depth)
        from centergrasp.rgb.pred_postprocessing import postprocess_predictions
        from centergrasp.rgb.data_structures import RgbdDataNp
        from centergrasp.graspnet.rgb_data import KINECT_HALF_PARAMS
        processed,_=postprocess_predictions(RgbdDataNp(rgb,depth,pred[0],None,None,None),pred[1],num_grasps=4000,use_icp=True,camera_params=KINECT_HALF_PARAMS)
        ns=definitions(repo/'centergrasp/graspnet/evaluation_runs.py',['grasp_panda_to_graspnet'],dict(np=np,Tuple=tuple))
        arrays=[]
        for obj in processed:
            if len(obj.grasp_poses):
                rotation,center=ns['grasp_panda_to_graspnet'](obj.grasp_poses);n=len(center)
                arrays.append(np.column_stack([np.linspace(1,1-.01*n,n,endpoint=False),np.full(n,.08),np.full(n,.02),np.full(n,.03),rotation,center,np.zeros(n)]))
        save_grasps(np.concatenate(arrays) if arrays else np.empty((0,17)))
        result.update(stage='dataset_inference',camera='kinect',protocol='Native RGB-D stride downsample/crop, SGDF, ICP, FCL collision and native grasp conversion')
    elif mid == 'motiongrasp':
        from models.motion_encoder import TemporalEncoderLayer
        temporal_forward=TemporalEncoderLayer.forward
        def compatible_forward(self,src,src_mask=None,src_key_padding_mask=None,is_causal=False):
            if is_causal and src_mask is None:raise ValueError('Causal attention requires the explicit upstream mask')
            return temporal_forward(self,src,src_mask,src_key_padding_mask)
        TemporalEncoderLayer.forward=compatible_forward
        from models.grasp_aligner import AttnEncoderLayer
        attn_forward=AttnEncoderLayer.forward
        def compatible_attn(self,src,src_mask=None,src_key_padding_mask=None,is_causal=False):
            if is_causal and src_mask is None:raise ValueError('Causal attention requires the explicit upstream mask')
            return attn_forward(self,src,src_mask,src_key_padding_mask)
        AttnEncoderLayer.forward=compatible_attn
        from models.graspnet import GraspNet
        from models.tracker import MotionTracker
        from models.motion_encoder import pred_decode
        from dataset.grasptracking_dataset import GraspTracking_Dataset,total_collate_fn
        dataset=GraspTracking_Dataset(args.dataset_root,camera=camera,split='test',num_points=15000,sequence_size=3,load_label=False)
        batch=total_collate_fn([dataset[0]])
        net=GraspNet().cuda().eval()
        net.load_state_dict(torch.load(ROOT/'checkpoints/graspnet_baseline/checkpoint-rs.tar',map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
        tracker=MotionTracker(device=0,is_training=False,nfr=5,feature_dim=128).cuda().eval()
        tracker.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint()['model_state_dict'].items()},strict=True)
        shapes=[]
        with torch.no_grad():
            for frame,data in enumerate(batch):
                end=net({k:v.cuda() for k,v in data.items()})
                pred=end['batch_grasp_preds'] if 'batch_grasp_preds' in end else pred_decode(end,remove_background=False)[0]
                tracked=tracker(frame,pred,end)
                shapes.append(list(pred.shape))
        memo=tracker.return_memo()
        assert torch.isfinite(memo).all();np.save(out/'trajectories.npy',memo.cpu().numpy())
        result.update(stage='real_sequence_tracking',frames=3,prediction_shapes=shapes,trajectory_shape=list(memo.shape),protocol='Original 3-frame tracking over all detector candidates; no GT-ranked query selection or tracking metric.')
    elif mid == 'graspness_modern':
        from models.graspnet import GraspNet,pred_decode
        import MinkowskiEngine as ME
        xyz=real_xyz(15000)
        model=GraspNet(is_training=False,backbone='resunet').cuda().eval()
        state=checkpoint();state=state.get('model_state_dict',state)
        # A deterministic view lattice became a persistent buffer after these
        # weights were published. It is generated by the same model constructor.
        state.setdefault('rotation.template_views',model.rotation.template_views.detach().cpu())
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        coords,feats=ME.utils.sparse_collate([xyz/.005],[np.ones_like(xyz)])
        coords,feats,_,inverse=ME.utils.sparse_quantize(coords,feats,return_index=True,return_inverse=True)
        data=dict(point_clouds=torch.from_numpy(xyz)[None].cuda(),coors=coords.cuda(),feats=feats.cuda(),quantize2original=inverse.cuda())
        with torch.no_grad():pred=pred_decode(model(data))
        save_grasps(pred[0]);result['stage']='real_input_pretrained_forward_decode'
    elif mid == 'rgb_matters':
        from rgbd_graspnet.net.rgb_normal_net import RGBNormalNet
        from rgbd_graspnet.data import GraspNetDataset
        from rgbd_graspnet.data.utils.convert import convert_grasp
        import cv2
        model=RGBNormalNet(num_layers=50,use_normal=True,normal_only=False).cuda().eval()
        model.load_state_dict(checkpoint()['net'],strict=True)
        dataset=GraspNetDataset(graspnet_root=args.dataset_root,use_normal=True,split='test_seen',camera=camera)
        rgb=dataset.rgb_transform(Image.open(directory/'rgb/0000.png'))[None].cuda()
        # Generate only this frame's normals with the author's gen_normals.py
        # settings, without writing into the user's dataset or a global cache.
        pcd=dataset.graspnet.loadScenePointCloud(100,camera,0,use_inpainting=True,use_mask=False,use_workspace=False)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(150))
        normal=((np.asarray(pcd.normals).reshape(720,1280,3)+1)/2*255).astype(np.uint8)
        normal=dataset.normal_transform(Image.fromarray(normal))[None].cuda()
        with torch.no_grad():pred=model(rgb,normal)[0].cpu().numpy().astype(np.float32)
        gg=convert_grasp(label=pred,scene_id=100,camera=camera,ann_id=0,graspnet_root=args.dataset_root,
            top_in_grid=5,top_in_map=1000,top_sample=200,topK=30,approach_dist=.05,collision_thresh=.001,
            empty_thresh=.10,nms_t=.04,nms_r=30,width_list=[.1],delta_depth_list=[-.02,0,.02],flip=False,device='cuda:0')
        save_grasps(gg.grasp_group_array);result['stage']='dataset_inference'
    elif mid == 'asgrasp':
        sys.path[:0]=[str(repo/p) for p in ('gsnet','gsnet/pointnet2','gsnet/utils','src/core_multilayers')]
        mod=importlib.import_module('infer_mvs_2layer_gsnet');mod.DEBUG_VIS=False
        cfg=SimpleNamespace(mixed_precision=False,valid_iters=16,hidden_dims=[128]*3,corr_implementation='reg',
            shared_backbone=False,corr_levels=4,corr_radius=4,n_downsample=2,context_norm='batch',slow_fast_gru=False,
            n_gru_layers=3,num_sample=96,depth_min=.2,depth_max=1.5,train_2layer=True,
            restore_ckpt=str(ROOT/'checkpoints/asgrasp/raftmvs.pth'),checkpoint_path=str(ROOT/'checkpoints/asgrasp/graspness.tar'),
            seed_feat_dim=512,graspness_threshold=0,collision_thresh=.01,voxel_size_cd=.01)
        model=mod.MVSGSNetEval(cfg)
        gg=model.infer(*[str(repo/'test_data'/f'00100_0000_{suffix}.png') for suffix in ('color','ir_l','ir_r')])
        if gg is None:raise ValueError('Upstream inference returned None; no successful output')
        save_grasps(gg.grasp_group_array);result.update(stage='author_stereo_sample_inference',protocol='RGB + left/right IR, supplied author sample; unavailable in standard G1B RGB-D alone.')
    elif mid == 'dreds':
        sys.path.insert(0,str(repo/'SwinDRNet'))
        from config import _C
        from networks.SwinDRNet import SwinDRNet
        cfg=_C.clone();cfg.merge_from_file(str(repo/'SwinDRNet/configs/swin_tiny_patch4_window7_224_lite.yaml'))
        model=SwinDRNet(cfg,img_size=224,num_classes=1).cuda().eval()
        state=checkpoint('model.pth');model.load_state_dict(state.get('model_state_dict',state.get('model',state)),strict=True)
        rgb=torch.from_numpy(np.array(Image.open(directory/'rgb/0000.png'))).permute(2,0,1)[None].cuda().float()/255.
        dep=torch.from_numpy(np.array(Image.open(directory/'depth/0000.png')).astype(np.float32))[None,None].cuda()/1000.
        with torch.no_grad():pred=model(torch.nn.functional.interpolate(rgb,(224,224)),torch.nn.functional.interpolate(dep,(224,224)))
        result.update(stage='depth_restoration_forward',output_shapes=[list(x.shape) for x in pred],protocol='Auxiliary depth network; not standalone grasp detector')
    elif mid == 'spahybgen':
        sys.path.insert(0,str(repo/'src'))
        from spahybgen.networks import load_network
        from spahybgen import inference
        from spahybgen.pipeline.grasp_optimization import GraspOptimization
        grid=np.load(repo/'assets/observations/scene_010_ann_0124_voxel.npz')['grid']
        if grid.ndim==3:grid=grid[None]
        if grid.ndim==4:grid=grid[0][None]
        if args.checkpoint:
            from spahybgen.networks import get_network
            net=get_network('unet',dict(voxel_discreteness=80,orientation='quat',augment=False)).cuda()
            net.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
        else:
            net=load_network(repo/'assets/trained_models/spahybgen_unet_64_voxel.pt',torch.device('cuda'),dict(voxel_discreteness=80,orientation='quat',augment=False))
        q,r,w=inference.process(*inference.predict(grid,net,torch.device('cuda')),gaussian_filter_sigma=0)
        inferred=np.vstack([grid,q[None],r,w[None]])
        optimizer=GraspOptimization(robot_name='robotiq2f',batch=4,penetration_mode='contact_penetration')
        trajectory,losses=optimizer.run_optimization(inferred,max_iter=3,tqdm_disable=True)
        torch.save(trajectory,out/'trajectory.pt');result.update(stage='voxel_network_and_three_optimization_steps',protocol='Author supplied voxel observation; 3 optimizer steps, no converged robotic result')
    else:
        raise ValueError('No recipe registered for '+mid)
    result['seconds']=time.monotonic()-start
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GRASPPANDA_RESULT='+json.dumps(result),flush=True)


if __name__=='__main__': main()
