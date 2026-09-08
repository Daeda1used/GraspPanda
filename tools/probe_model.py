"""One model probe in one process; all processes use the same shared interpreter."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('method')
parser.add_argument('--module')
parser.add_argument('--forward',action='store_true')
args=parser.parse_args()
method=next(m for m in json.loads((ROOT/'grasppanda/resources/methods.json').read_text()) if m['id']==args.method)
repo=ROOT/method['path']
os.chdir(repo)
sys.path[:0]=[str(repo),str(repo/'models'),str(repo/'utils'),str(repo/'knn'),str(repo/'pointnet2')]
if args.method=='contact_graspnet_g1b':
    sys.path.remove(str(repo/'utils'))
if args.method=='granet':sys.path.insert(1,str(repo/'dataset'))
if args.method=='graspbalance':
    sys.path[1:1]=[str(repo/x) for x in ('TrainModel','PointNet','KNN','DataProcessing','ModifiedNetTools')]
if args.method=='graspfast':
    sys.path[1:1]=[str(repo/x) for x in ('GraspFastModel','PointNet','KNN','DataProcessing')]
if args.method=='active_ngf':sys.path.insert(1,str(repo/'GraspNet'))
if args.method=='spahybgen':sys.path.insert(1,str(repo/'src'))
if args.method=='gfla':
    sys.path.insert(0,str(ROOT))
    from grasppanda.overlays import prepare_overlay
    sys.path.insert(0,str(prepare_overlay('gfla')))
    sys.path.insert(0,str(ROOT/'environments/extensions'))
    import grasp_nms_cpp
    sys.modules['models.my_grasp_nms.grasp_nms_cpp']=grasp_nms_cpp
if args.method=='centergrasp':
    sys.path.insert(0,str(ROOT))
    from grasppanda.overlays import prepare_overlay
    sys.path.insert(0,str(prepare_overlay('centergrasp')))
# Upstream modules sometimes parse global argv at import time.
sys.argv=[str(repo/'probe_model.py')]
if args.method=='economicgrasp':sys.argv+=['--dataset_root','/__synthetic_probe_no_dataset__','--camera','realsense']
import torch
torch.manual_seed(0)
torch.set_num_threads(4)
start=time.monotonic()
if args.method=='graspfast':
    sys.path.insert(0,str(ROOT))
    from grasppanda.graspfast import prepare
    module=prepare()
else:
    module=importlib.import_module(args.module or 'models.graspnet')
result={'method':args.method,'stage':'import','module':module.__name__,'module_file':module.__file__, 'python':sys.executable,'torch':torch.__version__}
if args.forward:
    if args.method in ['graspnet_baseline','scale_balanced_grasp','fgc_graspnet','pointnet2_upgrade','motiongrasp']:
        cls=module.GraspNet_MSCQ if args.method=='scale_balanced_grasp' else (module.FGC_graspnet if args.method=='fgc_graspnet' else module.GraspNet)
        model=cls(**({'is_training':False,'is_demo':True} if args.method=='fgc_graspnet' else {'is_training':False})).cuda().eval()
        inputs={'point_clouds':torch.rand(1,4096,3,device='cuda')}
        decoder=importlib.import_module('models.decode').pred_decode if args.method=='fgc_graspnet' else module.pred_decode
        with torch.no_grad():output=model(inputs);grasps=decoder(output)
        assert grasps and grasps[0].shape[-1]==17
        assert all(torch.isfinite(g).all() for g in grasps)
        result.update(stage='synthetic_forward_decode',grasp_shapes=[list(g.shape) for g in grasps])
    elif args.method in ['hggd','region_normalized_grasp']:
        model=module.AnchorGraspNet(ratio=8,in_dim=4).cuda().eval()
        with torch.no_grad():output=model(torch.randn(1,4,640,360,device='cuda'))
        result['stage']='component_forward'
        if args.method=='hggd':
            from models.localgraspnet import PointMultiGraspNet
            local=PointMultiGraspNet(info_size=3,k_cls=49).cuda().eval()
            with torch.no_grad():local_output=local(torch.randn(4,512,35,device='cuda'),torch.randn(4,3,device='cuda'))
            output={'anchor':output,'local':local_output}
            result['component_note']='Both branches executed separately; proposal grouping and final decoder not exercised.'
    elif args.method=='graspness':
        import MinkowskiEngine as ME
        model=module.GraspNet(is_training=False).cuda().eval()
        checkpoint=ROOT/'checkpoints/graspness/checkpoint-rs.tar'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state_dict'],strict=True)
        result['checkpoint']=str(checkpoint.relative_to(ROOT))
        # A synthetic table with a cuboid top and four sides, measured in metres.
        xyz=torch.rand(4096,3,device='cuda')
        xyz[:1024,:2]=(xyz[:1024,:2]-.5)*.5;xyz[:1024,2]=.75
        xyz[1024:,:2]=(xyz[1024:,:2]-.5)*.12;xyz[1024:,2]=.60+xyz[1024:,2]*.15
        for face in range(5):
            sl=slice(1024+face*614,min(4096,1024+(face+1)*614))
            if face==0:xyz[sl,2]=.60
            else:xyz[sl,(face-1)//2]=(-.06 if face%2 else .06)
        xyz=xyz.unsqueeze(0)
        def check_candidates(_module,_inputs,end):
            counts=((end['objectness_score'].argmax(1)==1)&(end['graspness_score'].squeeze(1)>module.GRASPNESS_THRESHOLD)).sum(1)
            if (counts==0).any():raise RuntimeError('Synthetic scene produced zero candidates; upstream FPS cannot safely process an empty set.')
        model.graspable.register_forward_hook(check_candidates)
        coords,feats=ME.utils.sparse_collate([xyz[0].cpu()/0.005],[torch.ones(4096,3)])
        coords,feats,_,inverse=ME.utils.sparse_quantize(coords,feats,return_index=True,return_inverse=True)
        inputs={'point_clouds':xyz,'coors':coords.cuda(),'feats':feats.cuda(),'quantize2original':inverse.cuda()}
        with torch.no_grad():output=model(inputs);grasps=module.pred_decode(output)
        assert all(torch.isfinite(g).all() for g in grasps)
        result.update(stage='synthetic_forward_decode',grasp_shapes=[list(g.shape) for g in grasps])
    elif args.method=='contact_graspnet_g1b':
        from types import SimpleNamespace
        model=module.ContactGraspNet(SimpleNamespace()).cuda().eval()
        with torch.no_grad():output=model(torch.rand(1,2048,3,device='cuda'))
        result['stage']='synthetic_forward'
    else:
        raise ValueError('No validated forward recipe for this method')
    def summarize(x):
        if isinstance(x,torch.Tensor):
            assert torch.isfinite(x).all(), 'Non-finite output'
            return list(x.shape)
        if isinstance(x,dict):return {k:summarize(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [summarize(v) for v in x]
        return str(type(x).__name__)
    result['output_shapes']=summarize(output)
    result['parameters']=sum(p.numel() for p in model.parameters())
    torch.cuda.synchronize()
result['seconds']=round(time.monotonic()-start,3)
print('GRASPPANDA_RESULT='+json.dumps(result))
