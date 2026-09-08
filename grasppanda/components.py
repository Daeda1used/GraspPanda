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
    ComponentSlot('backbone','view_estimator.backbone',('upstream','pointnet','pointnext','pointvector','pointmlp','sonata_ptv3'),
                  'Camera-frame point cloud [B,N,3], metres; N >= 1024.',
                  'Features [B,256,1024], coordinates [B,1024,3], and original-input fp2_inds.'),
    ComponentSlot('crop','grasp_generator.crop',('upstream','multiscale','cylinder'),
                  'Camera-frame seed points, scene points and proper approach rotations.',
                  'Features [B,256,1024,4] retaining the native four depth bins.'),
)


def slots(method):
    if method in ('graspnet_baseline','pointnet2_upgrade'):return BASELINE_SLOTS
    if method in ('hggd','region_normalized_grasp'):
        return (ComponentSlot('backbone','backbone',('upstream','native_resnet','convnextv2','repvit','mobilenetv4','dinov2','dinov3','vmamba'),
            'Native D,R,G,B image tensor [B,4,640,360], including the author axis convention and depth preprocessing.',
            'Five native feature lattices, strides 2/4/8/16/32 and channels 8/16/32/64/128; anchor heads and local refinement remain native.'),)
    if method=='finegrasp':return (
        ComponentSlot('backbone','backbone',('upstream','sonata_ptv3'),
            'Sparse camera XYZ and normal features; preserve voxel coordinate map and row order.',
            '512-channel sparse features for the native FineGrasp seed selector.'),
        ComponentSlot('crop','cy_groups',('upstream','native_cylinder'),
            'FineGrasp seed XYZ, 512-channel features and native approach rotations.',
            'Native 256-channel cylinder features per radius, consumed by multi-range attention.'))
    if method=='graspness':return (ComponentSlot('backbone','backbone',('upstream','pointnet','sparse_unet18','sonata_ptv3'),
        'Sparse RGB/constant features and voxel coordinates; retain the coordinate map and row order.',
        '512-channel sparse features, mapped to original input points by quantize2original.'),
        ComponentSlot('crop','crop',('upstream','cylinder','finegrasp'),
            'Graspable seed coordinates/features and approach rotations; native oriented cylinder queries.',
            '256-channel seed features preserving the native approach and depth decoder semantics.'))
    return ()


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
    changes=[]
    for slot in slots(method):
        from .module_options import unpack
        choice,options=unpack(selection.get(slot.name,'upstream'))
        if choice=='upstream':continue
        parent_name,attribute=slot.model_path.rsplit('.',1) if '.' in slot.model_path else ('',slot.model_path)
        parent=model.get_submodule(parent_name) if parent_name else model
        native=getattr(parent,attribute)
        if method in ('hggd','region_normalized_grasp') and choice in ('dinov2','dinov3'):
            from .modules.dino import DinoPyramid
            replacement=DinoPyramid(choice,**options)
        elif method in ('hggd','region_normalized_grasp') and choice == 'vmamba':
            from .modules.vmamba import VMambaPyramid
            replacement = VMambaPyramid(**options)
        elif method in ('hggd','region_normalized_grasp'):
            from .modules.image_pyramid import ImagePyramid,native_resnet
            replacement=native_resnet(native,**options) if choice=='native_resnet' else ImagePyramid(choice,**options)
        elif choice=='sonata_ptv3':
            from .modules.sonata import SonataBackbone,SparseSonataBackbone
            replacement=SparseSonataBackbone(model.seed_feature_dim,voxel_size,feature_channels=6 if method=='finegrasp' and model.use_normal else 3,**options) if method in ('graspness','finegrasp') else SonataBackbone(voxel_size,**options)
        elif method=='finegrasp' and choice=='native_cylinder':
            from torch import nn
            from .finegrasp import native_module
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
        elif method=='graspness' and choice=='pointnet':
            from .modules.sparse_pointnet import SparsePointNet
            replacement=SparsePointNet(model.seed_feature_dim,voxel_size)
        elif method=='graspness' and slot.name=='crop' and choice=='finegrasp':
            from .modules.finegrasp import FineGraspCrop
            replacement=FineGraspCrop(native,**options)
        elif slot.name=='crop' and choice=='cylinder':
            from .modules.cylinder import CylindricalAggregation
            replacement=CylindricalAggregation(native,'graspness' if method=='graspness' else 'baseline',**options)
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
        setattr(parent,attribute,replacement)
        changes.append(slot.model_path+'.')
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
    for prefix in changed_prefixes:
        module = model.get_submodule(prefix.rstrip('.'))
        if isinstance(module, DinoPyramid):
            record = module.initialize_pretrained()
            if record: pretrained[prefix.rstrip('.')] = record
    return dict(policy=policy, initialized=initialized, discarded=discarded, pretrained=pretrained,
                note='Replaced components use declared initialization; unchanged modules reuse the checkpoint.')
