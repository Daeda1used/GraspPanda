"""Bounded real-label training checks with explicit gradient and update evidence."""
import importlib
import itertools
from pathlib import Path
import sys
import time
from .jobs import digest


def rgb_matters(config,out,steps=3):
    """Native RGB/normal AVH prediction with real generated grasp targets."""
    import cv2
    import numpy as np
    import torch
    import open3d as o3d
    from .worker import prepare
    repo=prepare(config.method)
    from rgbd_graspnet.net.rgb_normal_net import RGBNormalNet
    from rgbd_graspnet.data import GraspNetDataset
    from rgbd_graspnet.data.utils.gen_label import batch_get_grid_label
    staging=out/'prepared'
    relative=Path(f'scene_{config.scene:04d}')/config.camera
    labels=staging/'labels'/relative;normals=staging/'normals'/relative
    labels.mkdir(parents=True);normals.mkdir(parents=True)
    dataset=GraspNetDataset(graspnet_root=config.dataset_root,label_root=str(staging/'labels'),normals_dir=str(staging/'normals'),camera=config.camera,split='train',use_normal=True)
    api=dataset.graspnet
    object_ids=api.getObjIds(config.scene)
    native_labels=api.loadGraspLabels(object_ids)
    collision=api.loadCollisionLabels(config.scene)
    target=batch_get_grid_label(config.scene,config.camera,config.frame,api,grasp_labels=native_labels,collision_labels=collision)
    positive=int(target.sum())
    if not positive:raise ValueError('No positive AVH labels in the selected real frame')
    label_path=labels/f'{config.frame:04d}.npy';np.save(label_path,target)
    del native_labels,collision,target
    cloud=api.loadScenePointCloud(config.scene,config.camera,config.frame,use_inpainting=True,use_mask=False,use_workspace=False)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(150))
    normal=((np.asarray(cloud.normals).reshape(720,1280,3)+1)/2*255).astype(np.uint8)
    normal_path=normals/f'{config.frame:04d}.png'
    if not cv2.imwrite(str(normal_path),normal):raise ValueError('Could not write native normals')
    rgb,_,target,normal=dataset[config.scene*256+config.frame]
    rgb,normal,target=(x[None].cuda() for x in (rgb,normal,target))
    model=RGBNormalNet(num_layers=50,use_normal=True,normal_only=False).cuda().train()
    state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['net'],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    criterion=torch.nn.MSELoss()
    losses=[];updates=[];start=time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction=model(rgb,normal);loss=criterion(prediction,target)
        if not torch.isfinite(loss):raise ValueError('Non-finite RGB Matters native MSE')
        loss.backward()
        params=[p for p in model.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Invalid RGB Matters gradients')
        selected=[p for p in params if torch.count_nonzero(p.grad)]
        if not selected:raise ValueError('All RGB Matters gradients are zero')
        before=selected[0].detach().clone()
        norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        # The pinned rs_rs_norm training configuration clips at 5.
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        delta=float((selected[0].detach()-before).norm())
        if not delta>0 or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Invalid RGB Matters parameter update')
        losses.append(dict(total=float(loss.detach()),components={'native_avh_mse':float(loss.detach())}))
        updates.append(dict(gradient_norm=norm,parameter_update_norm=delta,gradient_tensors=len(params)))
        print('LOSS',losses[-1],flush=True)
    torch.cuda.synchronize()
    torch.save(dict(net=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),total_iter=steps,config=config.to_dict()),out/'checkpoint.pt')
    root=Path(config.dataset_root)
    return dict(method=config.method,stage='real_label_avh_training',optimizer_steps=steps,losses=losses,updates=updates,
        seconds=time.monotonic()-start,camera=config.camera,scene=config.scene,frame=config.frame,positive_avh_targets=positive,
        checkpoint_sha256=digest(config.checkpoint),label_sha256=digest(label_path),normal_sha256=digest(normal_path),
        source_label_sha256={str(i):digest(root/'grasp_label'/f'{i:03d}_labels.npz') for i in object_ids},
        collision_sha256=digest(root/'collision_label'/f'scene_{config.scene:04d}'/'collision_labels.npz'),
        protocol='Native real-label AVH generation (friction threshold 0.1), 150-neighbor depth normals, RGBNormalNet-50 and original unweighted MSE objective; repeated real training frame, batch 1, no augmentation; Adam with reduced learning rate.',ap=None)


def hggd(config, out, steps=3, label_root=None):
    import numpy as np
    import torch
    from .worker import prepare
    from .components import configure_model,load_checkpoint
    from .optimization import build_optimizer,UpdateSchedule
    prepare('hggd')
    camera=importlib.import_module('dataset.config')
    intrinsic=camera.get_camera_intrinsic
    camera.camera=config.camera
    camera.get_camera_intrinsic=lambda camera=config.camera:intrinsic(camera)
    sys.argv=['train_graspnet.py','--joint-trainning','--optim','adamw']
    module=importlib.import_module('train_graspnet')
    args=module.parse_args()
    args.batch_size=2;args.step_cnt=1;args.pre_epochs=0;args.shift_epoch=0
    args.center_num=48;args.reg_b=5.;args.offset_d=1.;args.lr=config.learning_rate
    args.all_points_num=config.num_points
    labels=Path(label_root or config.label_root) if label_root or config.label_root else Path(config.dataset_root)/'HGGD_Preprocessed'/f'6dto2drefine_{config.camera}'
    required=labels/'6d_dataset'/f'scene_{config.scene}'/'grasp_labels'/f'{config.frame}_view.npz'
    if not required.exists():raise ValueError(f'HGGD preprocessed labels missing: {required}')
    dataset=module.GraspnetPointDataset(args.all_points_num,str(labels),config.dataset_root,[config.scene],
        noise=0,sigma=args.sigma,ratio=args.ratio,anchor_k=args.anchor_k,anchor_z=args.anchor_z,
        anchor_w=args.anchor_w,grasp_count=args.grasp_count,output_size=(640,360),random_rotate=False,random_zoom=False)
    from .image_augmentation import configure_dataset
    from .image_losses import ImageLosses
    configure_dataset(dataset, config)
    objectives=ImageLosses(importlib.import_module('models.losses'), config)
    # Native trainer skips the update at batch index 0. N+1 batches yield N updates.
    loader=torch.utils.data.DataLoader(dataset,batch_size=2,sampler=[config.frame]*((steps+1)*2),num_workers=0)
    anchor=module.AnchorGraspNet(in_dim=4,ratio=8,anchor_k=6)
    changed=configure_model(anchor,config.method,config.modules)
    anchor.cuda()
    local=module.PointMultiGraspNet(3,49).cuda()
    if not config.checkpoint:raise ValueError('A trained HGGD checkpoint is required for the bounded joint-training check')
    state=torch.load(config.checkpoint,map_location='cuda',weights_only=True)
    transfer=load_checkpoint(anchor,state['anchor'],changed,config.checkpoint_policy)
    local.load_state_dict({k:v for k,v in state['local'].items() if k.rsplit('.',1)[-1] not in ('total_ops','total_params')},strict=True)
    anchors={k:state[k].cuda() for k in ('gamma','beta')}
    parameters=itertools.chain(anchor.parameters(),local.parameters())
    optimizer=build_optimizer(parameters,config) if config.optimizer else module.get_optimizer(args,parameters)
    schedule=UpdateSchedule(optimizer,config.scheduler,steps) if config.scheduler else None
    losses=[];updates=[]
    anchor_loss=objectives.anchor;local_loss=objectives.local
    def measured_anchor(*a,**kw):
        value=anchor_loss(*a,**kw)
        losses.append(dict(anchor=float(value['loss'].detach()),**{k:float(v.detach()) for k,v in value['losses'].items()}))
        return value
    def measured_local(*a,**kw):
        value=local_loss(*a,**kw)
        losses[-1].update(local_classification=float(value[0].detach()),offset=float(value[1].detach()))
        losses[-1]['total']=losses[-1]['anchor']+sum(float(v.detach()) for v in value)
        if not all(np.isfinite(x) for x in losses[-1].values()):raise ValueError('Non-finite native training loss')
        print('LOSS',losses[-1],flush=True)
        return value
    module.compute_anchor_loss=measured_anchor;module.compute_multicls_loss=measured_local
    original_step=optimizer.step
    def measured_step(*a,**kw):
        record={'learning_rate':optimizer.param_groups[0]['lr']}
        snapshots={}
        branches=([('anchor',anchor)]+[('anchor.'+prefix[:-1],anchor.get_submodule(prefix[:-1])) for prefix in changed] if objectives.branch_enabled('anchor_') else [])
        if objectives.branch_enabled('local_'):branches.append(('local',local))
        for name,net in branches:
            params=[p for p in net.parameters() if p.grad is not None]
            if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError(f'Invalid {name} gradients')
            nonzero=[p for p in params if torch.count_nonzero(p.grad)]
            if not nonzero:raise ValueError(f'No nonzero {name} gradients')
            snapshots[name]=(nonzero[0],nonzero[0].detach().clone())
            record[name+'_gradient_tensors']=len(params)
            record[name+'_gradient_norm']=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        answer=original_step(*a,**kw)
        for name,(parameter,before) in snapshots.items():
            record[name+'_parameter_update_norm']=float((parameter.detach()-before).norm())
            if not record[name+'_parameter_update_norm']>0:raise ValueError(f'No parameter update in {name}')
        updates.append(record)
        if schedule:schedule.step()
        return answer
    optimizer.step=measured_step
    start=time.monotonic()
    module.train(0,anchor,local,loader,optimizer,anchors,args)
    if len(updates)!=steps or len(losses)!=steps+1 or not all('total' in x for x in losses):
        raise ValueError('Native trainer skipped requested joint optimization steps')
    torch.cuda.synchronize()
    torch.save({'anchor':anchor.state_dict(),'local':local.state_dict(),
                'gamma':anchors['gamma'],'beta':anchors['beta'],
                'optimizer_state_dict':optimizer.state_dict(),'training_steps':steps,
                **({'scheduler_state_dict':schedule.state_dict()} if schedule else {}),
                'epoch':0,'config':config.to_dict()},out/'checkpoint.pt')
    return dict(method='hggd',stage='real_label_joint_training',optimizer_steps=len(updates),
        modules=config.modules,checkpoint_transfer=transfer,loss_config=config.loss,augmentation=config.augmentation,
        optimizer_config=config.optimizer,scheduler_config=config.scheduler,optimizer_class=type(optimizer).__name__,
        losses=losses,updates=updates,seconds=time.monotonic()-start,
        checkpoint_sha256=digest(config.checkpoint),label_sha256=digest(required),camera=config.camera,
        scene=config.scene,frame=config.frame,learning_rate=config.learning_rate,ap=None,
        protocol='Original HGGD joint trainer and real preprocessed labels; repeated real frame, batch size 2 (native squeeze requires a batch axis); 48 centers; reduced learning rate. Registered loss and RGB-D observation controls are applied when configured.',
        loss_note='Loss components measured before the native trainer mutates the anchor-loss tensor in place. A few updates establish training viability, not convergence.')


def point_family(config, out, steps=3):
    """Use native scene labels/loaders/models/losses without loading unrelated scenes."""
    import copy
    import numpy as np
    import torch
    import scipy.io
    from .worker import prepare
    from .training_options import augment_sample,weighted_loss
    from .optimization import build_optimizer,UpdateSchedule
    repo=prepare(config.method)
    if config.method=='graspbalance':sys.path[:0]=[str(repo/p) for p in ('TrainModel','PointNet','KNN','DataProcessing','ModifiedNetTools')]
    sys.argv=['train.py','--dataset_root',config.dataset_root,'--camera',config.camera]
    economic=config.method=='economicgrasp'
    modern=config.method=='graspness_modern'
    fusion=config.method=='generalizing_grasp'
    balance=config.method=='graspbalance'
    graph=config.method=='granet'
    if graph:
        from .compat import legacy_dgl
        legacy_dgl()
    sparse=config.method in ('graspness','dograspnet','graspness_modern')
    fgc=config.method=='fgc_graspnet'
    dataset_module=importlib.import_module('mink_dataset' if fusion else ('DataProcessing.graspnet_dataset' if balance else ('dataset.graspnet_dataset_granet' if graph else 'dataset.graspnet_dataset')))
    module_name={'economicgrasp':'models.economicgrasp','dograspnet':'graspnet','fgc_graspnet':'models.FGC_graspnet','generalizing_grasp':'graspnet_sparseconv','graspbalance':'TrainModel.graspbalance','granet':'models.granet_pipeline'}.get(config.method,'models.graspnet')
    model_module=importlib.import_module(module_name)
    root=Path(config.dataset_root);scene=f'scene_{config.scene:04d}'
    directory=root/'scenes'/scene/config.camera
    meta=scipy.io.loadmat(directory/'meta'/f'{config.frame:04d}.mat')
    evidence={};labels={}
    if not economic:
        for obj in meta['cls_indexes'].flatten().astype(int):
            if obj==19 and not sparse:continue
            name=f'{obj-1:03d}'
            path=root/('grasp_label_simplified' if sparse else 'grasp_label')/(name+'_labels.npz')
            with np.load(path) as label:
                if sparse:labels[obj]=tuple(label[k].astype(np.float32) for k in ('points','width','scores'))
                elif graph:labels[obj]=tuple(label[k].astype(np.float32) for k in ('points','offsets','scores'))
                elif fgc:
                    extra=root/'FGC_label'/(name+'_labels.npz')
                    with np.load(extra) as scores:labels[obj]=(label['points'].astype(np.float32),label['offsets'].astype(np.float32),scores['new_scores'].astype(np.float32))
                    evidence['fgc_'+name]=digest(extra)
                else:
                    extra=root/'tolerance'/(name+'_tolerance.npy')
                    labels[obj]=tuple(label[k].astype(np.float32) for k in ('points','offsets','scores'))+(np.load(extra),)
                    evidence['tolerance_'+name]=digest(extra)
            evidence['labels_'+name]=digest(path)
    kwargs=dict(root=config.dataset_root,camera=config.camera,split='train',num_points=config.num_points,remove_outlier=True,augment=False,load_label=False)
    if not economic:kwargs['grasp_labels']=labels
    if sparse or economic or fusion:kwargs['voxel_size']=config.voxel_size
    if not sparse and not economic:kwargs['valid_obj_idxs']=list(labels)
    dataset_cls=getattr(dataset_module,'GraspNetDataset_fusion' if fusion else ('GraspPoseDataset' if balance else 'GraspNetDataset'))
    dataset=dataset_cls(**kwargs)
    if economic:
        path=root/'economic_grasp_label_300views'/(scene+'_labels.npz')
        dataset.grasp_labels[scene]=str(path);evidence['economic_labels']=digest(path)
    else:
        path=root/'collision_label'/scene/'collision_labels.npz'
        if not modern:
            with np.load(path) as collision:dataset.collision_labels[scene]={i:collision[f'arr_{i}'] for i in range(len(collision))}
        evidence['collision']=digest(path)
    dataset.load_label=True
    if config.method=='dograspnet':
        index=config.scene*256+config.frame
        requested=Path(dataset.graspnesspath[index])
        if not requested.exists():
            alternative=root/'graspness'/scene/config.camera/f'{config.frame:04d}.npy'
            if not alternative.exists():raise ValueError('DOGraspNet requires graspness_label/ or equivalent Graspness targets under graspness/')
            # The pinned generation algorithms differ only in imports, CLI defaults
            # and output folder; friction threshold, mask and normalization match.
            dataset.graspnesspath[index]=str(alternative)
            evidence['graspness_folder_alias']=digest(alternative)
    if graph:
        # Original generator writes only objectness_score/ under this staging
        # root. Inputs are read-only links, never generated into the dataset.
        staging=out/'prepared';staging.mkdir(exist_ok=True)
        for folder in ('scenes','grasp_label','collision_label'):
            link=staging/folder
            if not link.exists():link.symlink_to(root/folder,target_is_directory=True)
        generator=importlib.import_module('dataset.generate_objectness')
        generator.root=str(staging)
        generator.load_grasp_labels=lambda:(list(labels),{key:value[:3] for key,value in labels.items()})
        gen=generator.GenerateObjectness(config.scene,config.scene+1,10000,load_label=False,camera=config.camera)
        gen.collision_labels=dataset.collision_labels
        for name in ('depth_path','label_path','meta_path','scene_name','frame_id'):
            setattr(gen,name,[getattr(gen,name)[config.frame]])
        gen.generate_objectness()
        index=config.scene*256+config.frame
        for name,prefix in [('objectness_score','objectness_score_'),('objectness_sampled','objectness_sampled_')]:
            path=staging/'objectness_score'/scene/config.camera/(prefix+f'{config.frame:04d}.npy')
            getattr(dataset,name)[index]=str(path);evidence[name]=digest(path)
    if fusion:
        cloud=np.load(dataset.pcdpath[config.scene],allow_pickle=True).item()
        segmentation=np.load(dataset.labelpath[config.scene])
        if len(cloud['xyz'])!=len(segmentation):
            raise ValueError('Fused points and segmentation have different row counts. Rebuild matched fusion/segmentation files or select another intact training scene; the toolbox will not guess a correspondence.')
    data=dataset[config.scene if fusion else config.scene*256+config.frame]
    if fusion:
        evidence['fusion_points']=digest(root/'fusion_scenes'/scene/config.camera/'points.npy')
        evidence['fusion_segmentation']=digest(root/'fusion_scenes'/scene/config.camera/'seg.npy')
    collate=dataset_module.spconv_collate_fn if modern else (dataset_module.minkowski_collate_fn if sparse or fusion else dataset_module.collate_fn)
    cls=getattr(model_module,{'economicgrasp':'economicgrasp','fgc_graspnet':'FGC_graspnet','scale_balanced_grasp':'GraspNet_MSCQ','generalizing_grasp':'GraspNet_MSCQ','graspbalance':'GraspBalance','granet':'GraNet'}.get(config.method,'GraspNet'))
    model=cls(is_training=True,**({'is_demo':False} if fgc else ({'backbone':'resunet'} if modern else ({'batch_size':2} if graph else {})))).cuda().train()
    from .components import configure_model,load_checkpoint
    prefixes=configure_model(model,config.method,config.modules,config.voxel_size);model.cuda()
    transfer=None
    if config.checkpoint:
        state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
        state=state.get('model_state_dict',state)
        if modern:
            state={k.removeprefix('module.'):v for k,v in state.items()}
            state.setdefault('rotation.template_views',model.rotation.template_views.detach().cpu())
        transfer=load_checkpoint(model,state,prefixes,config.checkpoint_policy)
    if sparse or economic:
        threshold=model_module.cfgs.graspness_threshold if economic else model_module.GRASPNESS_THRESHOLD
        def guard(_module,_inputs,end):
            mask=(end['objectness_score'].argmax(1)==1)&(end['graspness_score'].squeeze(1)>threshold)
            if (mask.sum(1)==0).any():raise ValueError('Native graspable-point sampling would receive an empty set')
        model.graspable.register_forward_hook(guard)
    if fusion:
        # Change only the pinned module's import-time dataset-root argument.
        # All SDF loading and contact-loss mathematics remain verbatim.
        import ast,types
        source=repo/'utils/contact_point_loss.py'
        tree=ast.parse(source.read_text())
        calls=[node.value for node in tree.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='load_SDF']
        if len(calls)!=1:raise ValueError('Unexpected upstream SDF initialization; review the source pin')
        sdf_root=Path(config.sdf_root or config.dataset_root)
        calls[0].args=[ast.Constant(value=str(sdf_root))]
        module=types.ModuleType('contact_point_loss');module.__file__=str(source)
        sys.modules['contact_point_loss']=module
        exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),module.__dict__)
        for obj in labels:evidence['sdf_'+str(obj-1)]=digest(sdf_root/'models'/f'{obj-1:03d}'/'grid_sampled_sdf.npz')
    loss_module=importlib.import_module('models.loss_economicgrasp' if economic else ('TrainModel.loss' if balance else ('loss' if config.method=='dograspnet' or fusion else 'models.loss')))
    optimizer=build_optimizer(model.parameters(),config) if config.optimizer else torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    schedule=UpdateSchedule(optimizer,config.scheduler,steps) if config.scheduler else None
    def cuda(value):
        if graph and type(value).__module__.startswith('dgl.'):return value.to('cuda')
        if isinstance(value,torch.Tensor):return value.cuda()
        if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
        if isinstance(value,list):return [cuda(v) for v in value]
        return value
    losses=[];updates=[];start=time.monotonic()
    for step in range(steps):
        batch=cuda(collate([augment_sample(copy.deepcopy(data),dataset,config) for _ in range(2 if fusion or graph else 1)]))
        optimizer.zero_grad(set_to_none=True)
        output=model(batch)
        if economic:output['epoch']=0
        loss,output=loss_module.get_loss(output)
        loss,output=weighted_loss(loss,output,config)
        parts={k:float(v.detach()) for k,v in output.items() if 'loss' in k.lower() and isinstance(v,torch.Tensor) and v.numel()==1}
        if not torch.isfinite(loss) or not all(np.isfinite(x) for x in parts.values()):raise ValueError('Non-finite native loss')
        losses.append(dict(total=float(loss.detach()),components=parts))
        loss.backward()
        params=[p for p in model.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing or non-finite gradients')
        nonzero=[p for p in params if torch.count_nonzero(p.grad)]
        if not nonzero:raise ValueError('All gradients are zero')
        component_snapshots={}
        for prefix in prefixes:
            selected=[p for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None and torch.count_nonzero(p.grad)]
            if not selected:raise ValueError(f'No nonzero gradient reaches replacement component {prefix}')
            component_snapshots[prefix]=(selected[0],selected[0].detach().clone())
        before=nonzero[0].detach().clone()
        norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        used_lr=optimizer.param_groups[0]['lr']
        optimizer.step()
        update=float((nonzero[0].detach()-before).norm())
        if not update>0 or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('No valid parameter update')
        component_updates={name:float((p.detach()-old).norm()) for name,(p,old) in component_snapshots.items()}
        if not all(value>0 for value in component_updates.values()):raise ValueError('Replacement component did not update')
        updates.append(dict(gradient_tensors=len(params),gradient_norm=norm,parameter_update_norm=update,component_updates=component_updates,learning_rate=used_lr))
        if schedule:schedule.step()
        print('TRAINING_STEP',step+1,losses[-1],updates[-1],flush=True)
    torch.cuda.synchronize()
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                **({'scheduler_state_dict':schedule.state_dict()} if schedule else {}),
                'training_steps':steps,'epoch':0,'config':config.to_dict()},out/'checkpoint.pt')
    return dict(method=config.method,stage='real_label_training',modules=config.modules,augmentation=config.augmentation,loss_config=config.loss,checkpoint_transfer=transfer,optimizer_steps=len(updates),losses=losses,updates=updates,
        optimizer_config=config.optimizer,scheduler_config=config.scheduler,optimizer_class=type(optimizer).__name__,
        seconds=time.monotonic()-start,label_sha256=evidence,checkpoint_sha256=digest(config.checkpoint) if config.checkpoint else None,
        camera=config.camera,scene=config.scene,frame=config.frame,num_points=config.num_points,learning_rate=config.learning_rate,ap=None,
        protocol=('Repeated native fused training scene in table coordinates, original MSCQ and SDF contact losses; batch size 2 preserves native contact-loss batch axes.' if fusion else 'Repeated fixed real-label frame, native model/loss.')+(' ' if fusion else (' Batch size 2 retains the native VPS class axis.' if graph else ' Batch size 1. '))+('Configured augmentation. ' if config.augmentation else 'No augmentation. ')+'Loss and optimization settings are recorded in the result.',
        initialization='checkpoint' if config.checkpoint else 'random_constructor',
        limitation='The upstream GraspBalance driver references obsolete class names. This uses its actual GraspBalance detector and original loss with the native single-view loader; NcM augmentation and optional inference-time object balancing are not exercised.' if balance else None)


def rng(config,out,steps=3,label_root=None):
    """Native RNG objectives on real labels and fixed native heatmap proposals.

    The pinned repository has no released training driver/patch archive. This
    checks its actual losses using patches prepared by its own inference code;
    it does not claim reproduction of the unreleased training schedule.
    """
    from types import SimpleNamespace
    import numpy as np
    import torch
    from .worker import prepare
    from .components import configure_model,load_checkpoint
    from .optimization import build_optimizer,UpdateSchedule
    prepare('region_normalized_grasp')
    camera=importlib.import_module('dataset.config')
    intrinsic=camera.get_camera_intrinsic
    camera.camera=config.camera
    camera.get_camera_intrinsic=lambda camera=config.camera:intrinsic(camera)
    sys.argv=['demo.py','--center-num','48','--all-points-num',str(config.num_points),
              '--group-num','512','--input-w','640','--input-h','360','--embed-dim','256','--patch-size','64']
    demo=importlib.import_module('demo')
    demo.anchornet=demo.AnchorGraspNet(in_dim=4,ratio=8,anchor_k=6)
    changed=configure_model(demo.anchornet,config.method,config.modules)
    demo.anchornet.cuda().eval()
    demo.localnet=demo.PatchMultiGraspNet(49,theta_k_cls=6,feat_dim=256,anchor_w=60).cuda().eval()
    state=torch.load(config.checkpoint,map_location='cuda',weights_only=True)
    transfer=load_checkpoint(demo.anchornet,state['anchor'],changed,config.checkpoint_policy)
    localstate={k:v for k,v in state['local'].items() if k.rsplit('.',1)[-1] not in ('total_ops','total_params')}
    demo.localnet.load_state_dict(localstate,strict=True)
    demo.anchors={k:state[k].cuda() for k in ('gamma','beta')}
    labels=Path(label_root or config.label_root) if label_root or config.label_root else Path(config.dataset_root)/'HGGD_Preprocessed'/f'6dto2drefine_{config.camera}'
    path=labels/'6d_dataset'/f'scene_{config.scene}'/'grasp_labels'/f'{config.frame}_view.npz'
    if not path.is_file():raise ValueError('RNG requires author HGGD preprocessed real grasp targets')
    cls=importlib.import_module('dataset.graspnet_dataset').GraspnetAnchorDataset
    dataset=cls(str(labels),config.dataset_root,[config.scene],ratio=8,anchor_k=6,anchor_z=20,
                anchor_w=75,grasp_count=5000,sigma=10,noise=0,random_rotate=False,random_zoom=False)
    dataset.is_aug=False;dataset.aug=None
    from .image_augmentation import configure_dataset
    from .image_losses import ImageLosses
    configure_dataset(dataset, config)
    x,target,*_=dataset[config.frame]
    x=x.cuda()[None];target=[v.cuda()[None].repeat(2,*([1]*v.ndim)) for v in target]
    anchor,local=demo.anchornet,demo.localnet
    parameters=itertools.chain(anchor.parameters(),local.parameters())
    optimizer=build_optimizer(parameters,config) if config.optimizer else torch.optim.AdamW(parameters,lr=config.learning_rate)
    total_steps=steps+config.proposal_warmup_steps
    schedule=UpdateSchedule(optimizer,config.scheduler,total_steps) if config.scheduler else None
    losses_module=importlib.import_module('models.losses')
    objectives=ImageLosses(losses_module, config)
    losses=[];updates=[];start=time.monotonic()
    # A replacement image encoder starts without learned grasp proposals.
    # Fit the actual anchor targets before using its own native proposal path.
    anchor.train()
    for step in range(config.proposal_warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        pred,_=anchor(x.repeat(2,1,1,1))
        first=objectives.anchor(pred,target,reg_b=5)
        loss=first['loss']
        if not torch.isfinite(loss):raise ValueError('Non-finite RNG proposal warmup loss')
        loss.backward()
        parameters=[p for p in anchor.parameters() if p.grad is not None]
        if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Invalid RNG proposal warmup gradients')
        selected=next((p for p in parameters if p.grad.count_nonzero()),None)
        if selected is None:raise ValueError('No RNG proposal warmup gradient')
        before=selected.detach().clone()
        record={'stage':'Anchor warmup','learning_rate':optimizer.param_groups[0]['lr']}
        optimizer.step()
        record['anchor_parameter_update_norm']=float((selected.detach()-before).norm())
        if not record['anchor_parameter_update_norm']>0 or not all(torch.isfinite(p).all() for p in anchor.parameters()):
            raise ValueError('Invalid RNG proposal warmup update')
        losses.append(dict(stage='Anchor warmup',total=float(loss.detach()),
                           components={k:float(v.detach()) for k,v in first['losses'].items()}))
        updates.append(record)
        if schedule:schedule.step()
        print('ANCHOR_WARMUP',step+1,losses[-1],record,flush=True)
    anchor.eval()
    rgb=torch.from_numpy(dataset.cur_rgb.copy()).float().cuda()[None]/255.
    depth=torch.from_numpy(dataset.cur_depth.copy()).float().cuda()[None]
    helper=demo.PointCloudHelper(config.num_points)
    view,_,_=helper.to_scene_points(rgb,depth,include_rgb=True)
    xyz=helper.to_xyz_maps(depth)
    capture={}
    class PatchesPrepared(Exception):pass
    original_center=demo.center2dtopc
    def capture_centers(*a,**kw):
        result=original_center(*a,**kw);capture['centers']=result[0];return result
    def capture_patches(_module,inputs):
        capture['patches']=inputs[0].detach().clone();raise PatchesPrepared()
    demo.center2dtopc=capture_centers
    handle=demo.localnet.register_forward_pre_hook(capture_patches)
    try:
        demo.inference(view.squeeze(0),torch.cat([rgb.squeeze(0),xyz.squeeze(0)],0),x,rgb,depth,
                       use_heatmap=True,vis_heatmap=False,vis_grasp=False)
    except PatchesPrepared:pass
    finally:handle.remove();demo.center2dtopc=original_center
    if 'patches' not in capture:raise ValueError('Native heatmap produced no local training patches')
    pc=importlib.import_module('dataset.pc_dataset_tools')
    with np.load(path) as source:
        group_labels,total_labels=pc.get_center_group_label(capture['centers'],[dict(source)],500,dis=.02)
    counts=[len(v) for v in group_labels]
    if not len(total_labels):raise ValueError('No real labels within native 2 cm proposal neighborhoods. For a newly initialized image encoder, configure proposal_warmup_steps or initialize from its trained checkpoint.')
    losses_module=importlib.import_module('models.losses')
    get_info=importlib.import_module('models.localgraspnet').get_grasp_infos
    anchor,local=demo.anchornet.train(),demo.localnet.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred,_=anchor(x.repeat(2,1,1,1))
        first=objectives.anchor(pred,target,reg_b=5)
        _,pred,offset,theta_cls,theta_offset,width_reg=local(capture['patches'])
        theta=objectives.theta(theta_cls,theta_offset,width_reg,group_labels,anchor_w=60)
        cls_loss,offset_loss,positive=objectives.local(pred,offset,group_labels,
            get_info(theta_cls,theta_offset,width_reg),demo.anchors,args=SimpleNamespace(alpha=.02,offset_coef=1.),return_labels=True)
        if not positive.count_nonzero():raise ValueError('No positive native orientation labels; offset supervision would be untested')
        total=first['loss']+theta['loss']+cls_loss+offset_loss
        components={**{'anchor_'+k:float(v.detach()) for k,v in first['losses'].items()},
                    **{'local_theta_'+k:float(v.detach()) for k,v in theta['losses'].items()},
                    'local_orientation':float(cls_loss.detach()),'local_offset':float(offset_loss.detach())}
        if not torch.isfinite(total) or not all(np.isfinite(v) for v in components.values()):raise ValueError('Non-finite native RNG loss')
        total.backward();snapshots={};record={'learning_rate':optimizer.param_groups[0]['lr']}
        branches={}
        if objectives.branch_enabled('anchor_'):
            branches.update({'anchor':anchor,**{'anchor.'+prefix[:-1]:anchor.get_submodule(prefix[:-1]) for prefix in changed}})
        if objectives.branch_enabled('local_'):branches['local']=local
        for name,net,term in (('theta',local.theta_cls,'local_theta_classification'),('theta_offset',local.theta_offset,'local_theta'),
                              ('width',local.width_reg,'local_width'),('orientation',local.anchor_cls,'local_orientation'),('offset',local.offset_reg,'local_offset')):
            if objectives.enabled(term):branches[name]=net
        for name,net in branches.items():
            params=[p for p in net.parameters() if p.grad is not None]
            if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError(f'Invalid {name} gradients')
            nonzero=[p for p in params if p.grad.count_nonzero()]
            if not nonzero:raise ValueError(f'No nonzero {name} gradient')
            snapshots[name]=(nonzero[0],nonzero[0].detach().clone())
            record[name+'_gradient_norm']=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
        optimizer.step()
        for name,(parameter,before) in snapshots.items():
            record[name+'_parameter_update_norm']=float((parameter.detach()-before).norm())
            if not record[name+'_parameter_update_norm']>0:raise ValueError(f'No {name} update')
        if not all(torch.isfinite(p).all() for p in itertools.chain(anchor.parameters(),local.parameters())):raise ValueError('Non-finite updated parameter')
        losses.append(dict(total=float(total.detach()),components=components,positive_orientation_targets=int(positive.count_nonzero())))
        updates.append(record);print('TRAINING_STEP',step+1,losses[-1],record,flush=True)
        if schedule:schedule.step()
    torch.cuda.synchronize()
    torch.save({'anchor':anchor.state_dict(),'local':local.state_dict(),**demo.anchors,
                **({'scheduler_state_dict':schedule.state_dict()} if schedule else {}),
                'optimizer_state_dict':optimizer.state_dict(),'training_steps':steps,'epoch':0,'config':config.to_dict()},out/'checkpoint.pt')
    return dict(method=config.method,stage='real_label_native_objective_training',optimizer_steps=total_steps,proposal_warmup_steps=config.proposal_warmup_steps,losses=losses,updates=updates,
        modules=config.modules,checkpoint_transfer=transfer,loss_config=config.loss,augmentation=config.augmentation,
        optimizer_config=config.optimizer,scheduler_config=config.scheduler,optimizer_class=type(optimizer).__name__,
        camera=config.camera,scene=config.scene,frame=config.frame,seconds=time.monotonic()-start,checkpoint_sha256=digest(config.checkpoint),
        label_sha256=digest(path),patches=len(counts),labelled_patches=sum(n>0 for n in counts),local_labels=sum(counts),
        learning_rate=config.learning_rate,ap=None,
        protocol='Real HGGD labels through the RNG native anchor target generator and native heatmap/camera/grid-sample patches. Optional anchor-only warmup uses the same real targets and optimizer before preparing fixed proposals from that anchor; no teacher or substituted labels. Configured RGB-D observation augmentation is sampled once before preparing the fixed frame and local patches. Anchor batch 2, local batch up to 48; AdamW unless overridden; alpha 0.02 m, anchor width 60 mm, offset coefficient 1. Registered loss controls preserve native target generation and reductions.',
        limitation='The upstream training driver and preprocessed patch archive are unreleased. This validates the native objectives and every prediction head using dynamically prepared real patches; it is not the unreleased training schedule or convergence.')


def contact(config,out,steps=3):
    """Build one native contact-label sample outside the read-only dataset."""
    from types import SimpleNamespace
    import numpy as np
    import torch
    import h5py
    from .worker import prepare
    repo=prepare(config.method)
    sys.path.remove(str(repo/'utils'))
    preprocess=importlib.import_module('scripts.preprocess_g1b')
    g=preprocess.GraspNet1BLoader(config.dataset_root,camera=config.camera,split='train')
    labels=preprocess.load_scene_grasp_labels(g,config.scene,config.camera)
    collision=g.loadCollisionLabels(config.scene)
    prepared=out/'prepared';prepared.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(dataset_root=str(prepared),camera=config.camera,split='train',num_points=config.num_points,vis=False)
    print('Preparing one native real-label contact sample',flush=True)
    preprocess.process_scene(args,g,config.scene,config.frame,labels,collision)
    path=prepared/'scene_contacts'/config.camera/'train'/f'{config.scene:06d}_{config.frame:06d}.h5'
    with h5py.File(path) as f:
        contacts=len(f['scene_contact_points'])
        if contacts==0:raise ValueError('Native contact preprocessing produced no supervision')
    dataset_cls=importlib.import_module('data.g1b').GraspNet1B
    dataset=dataset_cls(SimpleNamespace(DATA_PATH=str(prepared),NUM_POSITIVE_CONTACTS=4000,CAMERA=config.camera,subset='train'))
    data={k:v[None].cuda() for k,v in dataset[0].items() if isinstance(v,torch.Tensor)}
    model=importlib.import_module('models.cgnet').ContactGraspNet(SimpleNamespace()).cuda().train()
    state=torch.load(config.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['base_model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.0005)
    losses=[];updates=[];start=time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred=model(data['pc'])
        loss,confidence,width,adds=model.get_loss(pred,data)
        components=dict(confidence=float(confidence.detach()),width=float(width.detach()),adds=float(adds.detach()))
        if not torch.isfinite(loss) or not all(np.isfinite(v) for v in components.values()):raise ValueError('Non-finite native ContactGraspNet loss')
        loss.backward()
        params=[p for p in model.parameters() if p.grad is not None]
        if not params or not all(torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing or non-finite contact gradients')
        selected=[p for p in params if p.grad.count_nonzero()]
        if not selected:raise ValueError('No nonzero contact gradient')
        before=selected[0].detach().clone()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.))
        optimizer.step()
        update=float((selected[0].detach()-before).norm())
        if not update>0 or not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('No valid contact parameter update')
        losses.append(dict(total=float(loss.detach()),components=components))
        updates.append(dict(gradient_tensors=len(params),gradient_norm_before_clip=norm,parameter_update_norm=update))
        print('TRAINING_STEP',step+1,losses[-1],updates[-1],flush=True)
    torch.cuda.synchronize()
    torch.save({'base_model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':0,'training_steps':steps,
                'config':config.to_dict()},out/'checkpoint.pt')
    return dict(method=config.method,stage='real_label_native_contact_training',optimizer_steps=steps,losses=losses,updates=updates,
        camera=config.camera,scene=config.scene,frame=config.frame,num_points=config.num_points,learning_rate=config.learning_rate,
        native_contact_labels=contacts,label_sha256=digest(path),checkpoint_sha256=digest(config.checkpoint),seconds=time.monotonic()-start,ap=None,
        protocol='Native G1B best-per-contact preprocessing from real grasp/collision labels and GT-bounded scene points; author centering and 4000-contact sampling; batch 1; native confidence/width/ADD-S loss; AdamW and native gradient clipping at 1. A bounded training check, not convergence.')
