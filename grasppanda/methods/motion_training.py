"""Bounded native MotionGrasp tracking updates over a real labelled sequence."""
import copy
import time


def run(config,out,steps=3):
    import numpy as np
    import torch
    from grasppanda.worker import prepare
    from grasppanda.weights import primary
    from grasppanda.jobs import digest
    prepare('motiongrasp')
    from models.motion_encoder import TemporalEncoderLayer,pred_decode
    from models.grasp_aligner import AttnEncoderLayer
    for cls in (TemporalEncoderLayer,AttnEncoderLayer):
        native=cls.forward
        def forward(self,src,src_mask=None,src_key_padding_mask=None,is_causal=False,_native=native):
            if is_causal and src_mask is None:raise ValueError('Causal attention requires the explicit native mask')
            return _native(self,src,src_mask,src_key_padding_mask)
        cls.forward=forward
    from models.graspnet import GraspNet
    from models.tracker import MotionTracker,merge_frame_end_points
    from models.loss import get_loss
    from dataset.grasptracking_dataset import GraspTracking_Dataset,total_collate_fn,load_grasp_labels
    ids,labels=load_grasp_labels(config.dataset_root)
    dataset=GraspTracking_Dataset(config.dataset_root,ids,labels,camera=config.camera,split='train',
        num_points=config.num_points,sequence_size=7,remove_outlier=True,augment=True,load_label=True)
    # The native loader begins each 15-frame block at frame 1 and provides
    # seven-frame windows. Keep its selection and actual label construction.
    matches=[i for i,(scene,frames) in enumerate(zip(dataset.scenename,dataset.frameid))
        if scene==f'scene_{config.scene:04d}' and frames[0]==config.frame+1]
    if not matches:raise ValueError('Native MotionGrasp windows require frame % 15 <= 8, with frame <= 248; actual sequence begins at frame + 1')
    batch=total_collate_fn([dataset[matches[0]]])
    detector=GraspNet().cuda().eval()
    detector_path=primary('graspnet_baseline',config.camera)
    detector.load_state_dict(torch.load(detector_path,map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
    tracker=MotionTracker(device=0,is_training=True,nfr=5,feature_dim=128).cuda().train()
    checkpoint=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
    tracker.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['model_state_dict'].items()},strict=True)
    optimizer=torch.optim.Adam(tracker.parameters(),lr=config.learning_rate)
    losses=[];updates=[];masks=[];start=time.monotonic()
    for frame,data in enumerate(batch):
        if frame>steps:break
        with torch.no_grad():
            end=detector({k:v.cuda() for k,v in data.items()})
            if 'batch_grasp_preds' not in end:
                pred,end=pred_decode(end,remove_background=False)
                end['batch_grasp_preds']=pred
        if frame==0:
            first=copy.deepcopy(end);tracker(0,end['batch_grasp_preds'],first);continue
        coarse,fine,mask,trans,rot,poses=tracker(frame,end['batch_grasp_preds'],end)
        if not mask.any():raise ValueError('No native training grasps survive object mask and NMS')
        loss,values=get_loss(merge_frame_end_points([first,end]),fine,coarse,mask,trans,rot,poses,frame,'6d',nfr=5)
        parts={key:float(value.detach()) for key,value in values.items() if 'loss' in key and isinstance(value,torch.Tensor) and value.numel()==1}
        if not torch.isfinite(loss) or not all(np.isfinite(v) for v in parts.values()):raise ValueError('Non-finite MotionGrasp native loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        params=[p for p in tracker.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Invalid native tracking gradients')
        snapshots={}
        for prefix in ('tracker.','aligner.'):
            selected=[p for key,p in tracker.named_parameters() if key.startswith(prefix) and p.grad is not None and torch.count_nonzero(p.grad)]
            if not selected:raise ValueError('No gradient in '+prefix)
            snapshots[prefix]=(selected[0],selected[0].detach().clone())
        norm=float(torch.nn.utils.clip_grad_norm_(tracker.parameters(),10))
        optimizer.step()
        delta={key:float((p.detach()-old).norm()) for key,(p,old) in snapshots.items()}
        if not all(v>0 for v in delta.values()) or not all(torch.isfinite(p).all() for p in tracker.parameters()):raise ValueError('Invalid native tracking parameter update')
        losses.append(dict(frame=int(data['frame_id'][0]),total=float(loss.detach()),components=parts))
        updates.append(dict(gradient_norm=norm,parameter_update_norms=delta));masks.append(int(mask.sum()))
        print('LOSS',losses[-1],flush=True)
    tracker.clear()
    if len(updates)!=steps:raise ValueError('MotionGrasp check supports at most six updates per seven-frame sequence')
    torch.save(dict(model_state_dict=tracker.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=0,config=config.to_dict()),out/'checkpoint.pt')
    return dict(method='motiongrasp',stage='real_sequence_tracking_training',camera=config.camera,scene=config.scene,frame=config.frame,
        optimizer_steps=len(updates),losses=losses,updates=updates,training_grasp_counts=masks,
        checkpoint_sha256=digest(config.checkpoint),detector_checkpoint_sha256=digest(detector_path),seconds=time.monotonic()-start,
        protocol='Original seven-frame dataset windows and real segmentation/camera-pose supervision; frozen author detector, native tracker, correspondence and pose losses, Adam and gradient clipping 10. First requested temporal updates of one sequence; feature dimension 128 matches the released tracker checkpoint. No tracking benchmark or full training convergence.',ap=None)
