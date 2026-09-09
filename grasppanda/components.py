"""Semantic component slots and explicit checkpoint-transfer policies."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentSlot:
    name: str
    model_path: str
    choices: tuple[str, ...]
    input_contract: str
    output_contract: str


BASELINE_SLOTS = (
    ComponentSlot('backbone','view_estimator.backbone',('upstream','pointnet','pointnext','pointvector','pointmeta','pointmlp','pointmamba','pointcloud_mamba','octformer','sonata_ptv3','point_transformer_v2','litept','pointcnnpp','flash3d','oacnns','kpconvx','utonia','concerto'),
                  'Camera-frame point cloud [B,N,3], metres; N >= 1024.',
                  'Features [B,256,1024], coordinates [B,1024,3], and original-input fp2_inds.'),
    ComponentSlot('crop','grasp_generator.crop',('upstream','multiscale','cylinder','reslfe_cylinder','kpconvx_cylinder'),
                  'Camera-frame seed points, scene points and proper approach rotations.',
                  'Features [B,256,1024,4] retaining the native four depth bins.'),
)


def slots(method):
    if method == 'spgrasp': return (
        ComponentSlot('backbone', 'image_encoder', ('upstream', 'hiera'),
            'Letterboxed, ImageNet-normalized RGB sequence [T,3,R,R].',
            'Native Hiera feature pyramid for prompt decoding and temporal memory.'),
        ComponentSlot('memory', 'memory_attention', ('upstream', 'temporal'),
            'Native image features, first-frame object prompts and prior frame memories.',
            'Conditioned features for five planar grasp and semantic output channels.'))
    if method == 'scale_balanced_grasp': return (
        BASELINE_SLOTS[0],
        ComponentSlot('crop', 'grasp_generator', ('upstream', 'native_mscq'),
            'Native MSCQ endpoints: scene XYZ, seed XYZ/features, approach rotations and training labels.',
            'Native endpoint dictionary; four [B,256,1024,4] branch features feed unchanged scale fusion, seed gate and grasp heads.'))
    if method == 'economicgrasp': return (
        ComponentSlot('backbone', 'backbone', ('upstream', 'native_tdunet', 'pointnet', 'sonata_ptv3', 'point_transformer_v2','litept','pointcnnpp','flash3d','oacnns','kpconvx','utonia','concerto'),
            'Three constant features and quantized camera XYZ; retain the sparse coordinate map.',
            '512-channel sparse features in the input sparse row order, before quantize2original.'),
        ComponentSlot('crop', 'cy_group', ('upstream', 'native_cylinder', 'cylinder', 'reslfe_cylinder','kpconvx_cylinder'),
            'Camera-frame seed XYZ in metres, native cylinder grouping and approach rotations.',
            '256-channel seed features for the native interactive grasp head.'),
        ComponentSlot('head', 'grasp_head', ('upstream', 'native_interactive'),
            '256-channel grouped seed features; angle, depth, width and score interact independently per seed.',
            'Native angle/depth classes including invalid bins, six score classes and one scaled width.'))
    if method == 'gtg2': return (
        ComponentSlot('backbone', 'block', ('upstream', 'gtg_sage', 'gtg_gatv2'),
            'Candidate-local XYZ and inside/outside flags with undirected k-nearest graph edges.',
            'One grasp-quality score per candidate graph; ensemble members share the encoder contract.'),
        ComponentSlot('crop', 'graph', ('upstream', 'grasp_graph'),
            'GPG candidates and calibrated camera-frame workspace points in metres.',
            'Inside/outside point sets, explicit sampling caps and graph features.'))
    if method in ('graspnet_baseline','pointnet2_upgrade'):return BASELINE_SLOTS
    if method in ('hggd','region_normalized_grasp'):
        return (ComponentSlot('backbone','backbone',('upstream','native_resnet','convnextv2','repvit','mobilenetv4','dinov2','dinov3','vmamba','rala'),
            'Native D,R,G,B image tensor [B,4,640,360], including the author axis convention and depth preprocessing.',
            'Five native feature lattices, strides 2/4/8/16/32 and channels 8/16/32/64/128; anchor heads and local refinement remain native.'),)
    if method=='finegrasp':return (
        ComponentSlot('backbone','backbone',('upstream','sonata_ptv3','point_transformer_v2','litept','pointcnnpp','flash3d','oacnns','kpconvx','utonia','concerto'),
            'Sparse camera XYZ and normal features; preserve voxel coordinate map and row order.',
            '512-channel sparse features for the native FineGrasp seed selector.'),
        ComponentSlot('crop','cy_groups',('upstream','native_cylinder','kpconvx_cylinder'),
            'FineGrasp seed XYZ, 512-channel features and native approach rotations.',
            'Native 256-channel cylinder features per radius, consumed by multi-range attention.'))
    if method=='graspness':return (ComponentSlot('backbone','backbone',('upstream','pointnet','sparse_unet18','sonata_ptv3','point_transformer_v2','litept','pointcnnpp','flash3d','oacnns','kpconvx','utonia','concerto'),
        'Sparse RGB/constant features and voxel coordinates; retain the coordinate map and row order.',
        '512-channel sparse features, mapped to original input points by quantize2original.'),
        ComponentSlot('crop','crop',('upstream','cylinder','finegrasp','reslfe_cylinder','kpconvx_cylinder'),
            'Graspable seed coordinates/features and approach rotations; native oriented cylinder queries.',
            '256-channel seed features preserving the native approach and depth decoder semantics.'))
    return ()


def requires_scene_batch(config):
    """Encoders whose coarse batch normalization needs multiple training scenes."""
    from .module_options import unpack
    return unpack(config.modules.get('backbone', 'upstream'))[0] in (
        'pointcloud_mamba', 'point_transformer_v2', 'litept', 'oacnns', 'kpconvx')


def validate_selection(method,selection,checkpoint_policy='strict'):
    if not isinstance(selection,dict):raise ValueError('modules must be a mapping from slot name to implementation')
    available={s.name:s for s in slots(method)}
    for name,value in selection.items():
        from .module_options import unpack,validate_options
        if name not in available:raise ValueError(f'{method} has no registered {name} slot')
        choice,options=unpack(value)
        if choice not in available[name].choices:raise ValueError(f'{name} must be one of {available[name].choices}')
        validate_options(method,name,choice,options)
    if checkpoint_policy not in ('strict','reuse_unchanged'):
        raise ValueError('checkpoint_policy must be strict or reuse_unchanged')
    return selection


def configure_model(model,method,selection,voxel_size=.005):
    """Replace registered submodules only; return the exact changed state prefixes."""
    validate_selection(method,selection)
    if method == 'spgrasp':
        raise ValueError('SPGrasp constructs its image encoder and temporal memory together from the selected architecture')
    if method == 'gtg2':
        raise ValueError('GtG2 uses GraphRegressor with resolved graph options; its crop configures data construction, not a network submodule')
    changes=[]
    for slot in slots(method):
        from .module_options import unpack, SEED_INTERACTION_FIELDS
        choice,options=unpack(selection.get(slot.name,'upstream'))
        options = {key:value for key,value in options.items() if key not in SEED_INTERACTION_FIELDS}
        sampling_options = {key: options.pop(key) for key in ('seed_sampling', 'stage_sampling') if key in options}
        if choice=='upstream':continue
        if method == 'scale_balanced_grasp' and slot.name == 'crop':
            from .modules.mscq import configure
            changes.extend(configure(model.grasp_generator, options))
            continue
        parent_name,attribute=slot.model_path.rsplit('.',1) if '.' in slot.model_path else ('',slot.model_path)
        parent=model.get_submodule(parent_name) if parent_name else model
        native=getattr(parent,attribute)
        if method == 'economicgrasp' and choice == 'native_tdunet':
            from .modules.economic import tdunet
            replacement = tdunet(native, **options)
        elif method == 'economicgrasp' and choice == 'native_cylinder':
            from .modules.economic import cylinder
            replacement = cylinder(native, **options)
        elif method == 'economicgrasp' and choice == 'native_interactive':
            from .modules.economic import head
            replacement = head(native, **options)
        elif method in ('hggd','region_normalized_grasp') and choice in ('dinov2','dinov3'):
            from .modules.dino import DinoPyramid
            replacement=DinoPyramid(choice,**options)
        elif method in ('hggd','region_normalized_grasp') and choice == 'rala':
            from .modules.rala import RALAPyramid
            replacement = RALAPyramid(**options)
        elif method in ('hggd','region_normalized_grasp') and choice == 'vmamba':
            from .modules.vmamba import VMambaPyramid
            replacement = VMambaPyramid(**options)
        elif method in ('hggd','region_normalized_grasp'):
            from .modules.image_pyramid import ImagePyramid,native_resnet
            replacement=native_resnet(native,**options) if choice=='native_resnet' else ImagePyramid(choice,**options)
        elif choice in ('utonia', 'concerto'):
            from .modules.foundation import FoundationBackbone, SparseFoundationBackbone
            replacement = (SparseFoundationBackbone(choice, model.seed_feature_dim, voxel_size, **options)
                           if method in ('graspness', 'finegrasp', 'economicgrasp')
                           else FoundationBackbone(choice, **options))
        elif choice=='kpconvx':
            from .modules.kpconvx import KPConvXBackbone, SparseKPConvXBackbone
            replacement = (SparseKPConvXBackbone(model.seed_feature_dim, voxel_size,
                feature_channels=6 if method=='finegrasp' and model.use_normal else 3, **options)
                if method in ('graspness','finegrasp','economicgrasp') else KPConvXBackbone(**options))
        elif choice=='oacnns':
            from .modules.oacnns import OACNNBackbone, SparseOACNNBackbone
            replacement = (SparseOACNNBackbone(model.seed_feature_dim, voxel_size,
                feature_channels=6 if method=='finegrasp' and model.use_normal else 3, **options)
                if method in ('graspness','finegrasp','economicgrasp') else OACNNBackbone(voxel_size, **options))
        elif choice=='flash3d':
            from .modules.flash3d import Flash3DBackbone, SparseFlash3DBackbone
            replacement = (SparseFlash3DBackbone(model.seed_feature_dim, voxel_size,
                feature_channels=6 if method=='finegrasp' and model.use_normal else 3, **options)
                if method in ('graspness','finegrasp','economicgrasp') else Flash3DBackbone(**options))
        elif choice=='pointcnnpp':
            from .modules.pointcnnpp import PointCNNBackbone, SparsePointCNNBackbone
            replacement = (SparsePointCNNBackbone(model.seed_feature_dim, voxel_size,
                feature_channels=6 if method=='finegrasp' and model.use_normal else 3, **options)
                if method in ('graspness','finegrasp','economicgrasp') else PointCNNBackbone(**options))
        elif choice=='litept':
            from .modules.litept import LitePTBackbone, SparseLitePTBackbone
            replacement = (SparseLitePTBackbone(model.seed_feature_dim, voxel_size,
                feature_channels=6 if method=='finegrasp' and model.use_normal else 3, **options)
                if method in ('graspness','finegrasp','economicgrasp') else LitePTBackbone(voxel_size, **options))
        elif choice=='point_transformer_v2':
            from .modules.ptv2 import PTv2Backbone, SparsePTv2Backbone
            replacement = (SparsePTv2Backbone(model.seed_feature_dim, voxel_size,
                feature_channels=6 if method=='finegrasp' and model.use_normal else 3, **options)
                if method in ('graspness','finegrasp','economicgrasp') else PTv2Backbone(**options))
        elif choice=='sonata_ptv3':
            from .modules.sonata import SonataBackbone,SparseSonataBackbone
            replacement=SparseSonataBackbone(model.seed_feature_dim,voxel_size,feature_channels=6 if method=='finegrasp' and model.use_normal else 3,**options) if method in ('graspness','finegrasp','economicgrasp') else SonataBackbone(voxel_size,**options)
        elif method=='finegrasp' and choice=='native_cylinder':
            from torch import nn
            from grasppanda.methods.finegrasp import native_module
            source=native_module()
            from .modules.finegrasp import configure_fusion
            if configure_fusion(model.fuse_multi_scale, {k:v for k,v in options.items() if k.startswith('fusion_')}):
                changes.append('fuse_multi_scale.transformer.')
            factors=options.get('radius_factors',model.cylinder_groups)
            radius=options.get('radius',model.cylinder_radius)
            replacement=nn.ModuleList(source.CylinderGroup(nsample=options.get('nsample',16),
                seed_feature_dim=model.seed_feature_dim,cylinder_radius=radius*factor) for factor in factors)
        elif method=='graspness' and choice=='sparse_unet18':
            from models.backbone_resunet14 import MinkUNet18D
            replacement=MinkUNet18D(in_channels=3,out_channels=model.seed_feature_dim,D=3)
        elif method in ('graspness','economicgrasp') and choice=='pointnet':
            from .modules.sparse_pointnet import SparsePointNet
            replacement=SparsePointNet(model.seed_feature_dim,voxel_size)
        elif method=='graspness' and slot.name=='crop' and choice=='finegrasp':
            from .modules.finegrasp import FineGraspCrop
            replacement=FineGraspCrop(native,**options)
        elif slot.name=='crop' and choice=='kpconvx_cylinder':
            from .modules.kpconvx_cylinder import KPConvXCylinder
            if method=='finegrasp':
                import copy
                from torch import nn
                from .modules.finegrasp import configure_fusion
                fusion={k:v for k,v in options.items() if k.startswith('fusion_')}
                options={k:v for k,v in options.items() if k not in fusion}
                if configure_fusion(model.fuse_multi_scale,fusion):changes.append('fuse_multi_scale.transformer.')
                factors=options.pop('radius_factors',model.cylinder_groups)
                radius=options.pop('radius',model.cylinder_radius)
                prototype=copy.deepcopy(native[0]);prototype.grouper.radius=radius
                replacement=nn.ModuleList(KPConvXCylinder(prototype,'graspness',radius_factors=[factor],**options) for factor in factors)
            else:
                replacement=KPConvXCylinder(native,'graspness' if method in ('graspness','economicgrasp') else 'baseline',**options)
        elif slot.name=='crop' and choice=='reslfe_cylinder':
            from .modules.deepla import ResLFECylinder
            replacement=ResLFECylinder(native,'graspness' if method in ('graspness','economicgrasp') else 'baseline',**options)
        elif slot.name=='crop' and choice=='cylinder':
            from .modules.cylinder import CylindricalAggregation
            replacement=CylindricalAggregation(native,'graspness' if method in ('graspness','economicgrasp') else 'baseline',**options)
        elif choice=='octformer':
            from .modules.octformer import OctFormerBackbone
            replacement=OctFormerBackbone(**options)
        elif choice=='pointcloud_mamba':
            from .modules.pcm import PCMBackbone
            replacement=PCMBackbone(**options)
        elif choice=='pointmamba':
            from .modules.pointmamba import PointMambaBackbone
            replacement=PointMambaBackbone(**options)
        elif choice=='pointmeta':
            from .modules.pointmeta import PointMetaBackbone
            replacement=PointMetaBackbone(**options)
        elif choice=='pointvector':
            from .modules.pointvector import PointVectorBackbone
            replacement=PointVectorBackbone(**options)
        elif choice=='pointnext':
            from .modules.pointnext import PointNeXtBackbone
            replacement=PointNeXtBackbone(**options)
        elif choice=='pointmlp':
            from .modules.pointmlp import PointMLPBackbone
            replacement=PointMLPBackbone(**options)
        elif slot.name=='backbone':
            from .modules.pointnet import PointNetBackbone
            replacement=PointNetBackbone(**options)
        elif slot.name=='crop':
            from .modules.multiscale import MultiScaleCrop
            replacement=MultiScaleCrop(native,**options)
        else:raise ValueError(slot.name)
        if sampling_options:
            from .modules.sampling import configure_sampling
            configure_sampling(replacement, sampling_options)
        setattr(parent,attribute,replacement)
        changes.append(slot.model_path+'.')
    for slot in slots(method):
        if slot.name != 'crop': continue
        _, options = unpack(selection.get(slot.name, 'upstream'))
        if options.get('seed_interaction', 'none') == 'gaussian':
            from .modules.seed_interaction import install
            install(model.get_submodule(slot.model_path), **{
                key.removeprefix('interaction_'):value for key,value in options.items()
                if key.startswith('interaction_')})
            if slot.model_path+'.' not in changes:
                changes.append(slot.model_path+'.seed_interaction.')
    return changes


def load_checkpoint(model,state,changed_prefixes=(),policy='strict'):
    """Never hide unexpected mismatches outside explicitly replaced components."""
    if policy=='strict' or not changed_prefixes:
        model.load_state_dict(state,strict=True)
        return dict(policy='strict',initialized=[],discarded=[])
    if policy!='reuse_unchanged':raise ValueError('Unknown checkpoint policy')
    current=model.state_dict()
    selected=lambda key:any(key.startswith(prefix) for prefix in changed_prefixes)
    initialized=sorted(k for k in current if selected(k))
    discarded=sorted(k for k in state if selected(k))
    keep={k:v for k,v in state.items() if not selected(k)}
    expected={k for k in current if not selected(k)}
    if set(keep)!=expected:
        raise ValueError(f'Unchanged modules do not match: missing={sorted(expected-set(keep))}, unexpected={sorted(set(keep)-expected)}')
    for key in expected:
        if keep[key].shape!=current[key].shape:raise ValueError(f'Unchanged parameter shape mismatch: {key}')
    keep.update({k:current[k] for k in initialized})
    model.load_state_dict(keep,strict=True)
    pretrained = {}
    from .modules.dino import DinoPyramid
    from .modules.foundation import FoundationBackbone, SparseFoundationBackbone
    for prefix in changed_prefixes:
        module = model.get_submodule(prefix.rstrip('.'))
        if isinstance(module, (DinoPyramid, FoundationBackbone, SparseFoundationBackbone)):
            record = module.initialize_pretrained()
            if record: pretrained[prefix.rstrip('.')] = record
    return dict(policy=policy, initialized=initialized, discarded=discarded, pretrained=pretrained,
                note='Replaced components use declared initialization; unchanged modules reuse the checkpoint.')
