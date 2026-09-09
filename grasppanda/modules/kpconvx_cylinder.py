"""Native kernel-point blocks within independently queried grasp cylinders."""
from contextlib import nullcontext
import copy
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .cylinder import CylindricalAggregation
from .kpconvx import native_module
from .kpconvx_cylinder_options import resolve
from .ptv2 import recompute_buffers


class CylinderGroupNorm(nn.GroupNorm):
    """Apply the author's packed GroupNorm separately to each cylinder."""
    def __init__(self,norm,nsample):
        super().__init__(norm.num_groups,norm.num_channels,norm.eps,norm.affine)
        self.load_state_dict(norm.state_dict());self.nsample=nsample

    def forward(self,x):
        if x.ndim!=3 or x.shape[:2]!=(1,self.num_channels) or x.shape[2]%self.nsample:
            raise ValueError('Cylinder GroupNorm requires complete packed neighborhoods')
        packed=x.reshape(self.num_channels,-1,self.nsample).transpose(0,1)
        return super().forward(packed).transpose(0,1).reshape_as(x)


def local_neighbors(coordinates,count):
    """Stable KNN within each cylinder; repeated native query samples remain."""
    with torch.no_grad():
        distances=torch.cdist(coordinates,coordinates,compute_mode='donot_use_mm_for_euclid_dist')
        indices=distances.argsort(dim=-1,stable=True)[...,:count]
        offsets=torch.arange(len(coordinates),device=coordinates.device)[:,None,None]*coordinates.shape[1]
        return (indices+offsets).reshape(-1,count).contiguous()


def activation(name):
    return {'leaky_relu':lambda:nn.LeakyReLU(.1),'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}[name]()


class PointwiseFusion(nn.Module):
    def __init__(self,width,nonlinear):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(width,256),nn.LayerNorm(256),activation(nonlinear))

    def forward(self,x):
        return self.layers(x.movedim(1,-1)).movedim(-1,1).contiguous()


class KernelNeighborhood(nn.Module):
    def __init__(self,in_channels,**options):
        super().__init__()
        from ..module_options import validate_options
        validate_options('graspnet_baseline','crop','kpconvx_cylinder',options)
        self.options=resolve(options);p=self.options
        native_module();native=sys.modules['_grasppanda_kpconvx.utils.kpnext_blocks']
        generic=sys.modules['_grasppanda_kpconvx.utils.generic_blocks'];widths=p['channels']
        self.embedding=generic.UnaryBlock(in_channels,widths[0],norm_type=p['normalization'],
            bn_momentum=p['bn_momentum'],activation=activation(p['activation']))
        self.blocks=nn.ModuleList()
        for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
            groups=p['attention_groups'][i] if p['block']=='kpconvx' else 0
            block=native.KPNextMultiShortcutBlock(a,b,p['shell_sizes'],p['kernel_radius'][i],p['kernel_sigma'][i],
                attention_groups=groups,attention_act=p['attention_activation'],mod_grp_norm=p['modulation_norm'],
                expansion=p['expansion'][i],drop_path_p=p['drop_path'][i],layer_scale_init_v=p['layer_scale'][i],
                use_upcut=p['use_upcut'],influence_mode=p['influence'],norm_type=p['normalization'],
                bn_momentum=p['bn_momentum'],activation=activation(p['activation']))
            if groups:
                block.conv.modulation_scope='point' if p['modulation_scope']=='point' else 'native'
                if not p['modulation_norm']:block.conv.grpnorm=nn.Identity()
                elif p['modulation_scope']=='cylinder':block.conv.grpnorm=CylinderGroupNorm(block.conv.grpnorm,p['nsample'])
            self.blocks.append(block)
        self.projection=generic.UnaryBlock(widths[-1],256,norm_type=p['normalization'],
            bn_momentum=p['bn_momentum'],activation=activation(p['activation']))
        # Native feature GroupNorm flattens all packed points; preserve its
        # per-group affine parameters while isolating individual cylinders.
        for module in list(self.modules()):
            if isinstance(module,generic.GroupNormBlock):
                module.norm=CylinderGroupNorm(module.norm,p['nsample'])

    def encode_chunk(self,grouped):
        cylinders,count,channels=grouped.shape
        coordinates=grouped[...,:3].contiguous();xyz=coordinates.reshape(-1,3)
        neighbors=local_neighbors(coordinates,max(self.options['local_neighbors']))
        lengths=torch.full((cylinders,),count,dtype=torch.long,device=grouped.device)
        feature=self.embedding(grouped.reshape(-1,channels));upcut=None
        for block,k in zip(self.blocks,self.options['local_neighbors']):
            feature,upcut=block(xyz,xyz,feature,neighbors[:,:k].contiguous(),lengths,upcut=upcut)
        return self.projection(feature).reshape(cylinders,count,256)

    def forward(self,grouped):
        if grouped.ndim!=4 or grouped.shape[-1]!=self.options['nsample'] or grouped.shape[1]<3:
            raise ValueError('KPConvX cylinder expects grouped XYZ/features [B,C,M,nsample]')
        if grouped.dtype!=torch.float32 or not grouped.is_cuda:
            raise ValueError('KPConvX cylinder requires CUDA float32 grouped inputs')
        batch,_,seeds,count=grouped.shape
        if not batch or not seeds:raise ValueError('KPConvX cylinder requires nonempty neighborhoods')
        packed=grouped.permute(0,2,3,1).reshape(batch*seeds,count,-1)
        size=self.options['chunk_size'] or len(packed);outputs=[]
        for chunk in packed.split(size):
            if self.training and torch.is_grad_enabled() and self.options['checkpoint']:
                output=checkpoint(self.encode_chunk,chunk,use_reentrant=False,
                    context_fn=lambda:(nullcontext(),recompute_buffers(self)))
            else:output=self.encode_chunk(chunk)
            outputs.append(output)
        return torch.cat(outputs).reshape(batch,seeds,count,256).permute(0,3,1,2).contiguous()


class KPConvXCylinder(CylindricalAggregation):
    def __init__(self,native,protocol,**options):
        nn.Module.__init__(self)
        from ..module_options import validate_options
        validate_options('graspnet_baseline','crop','kpconvx_cylinder',options)
        p=resolve(options);self.protocol=protocol;self.pooling=p['pooling']
        queries=native.groupers if protocol=='baseline' else [native.grouper]
        self.depths=len(queries);self.groups=nn.ModuleList()
        for factor in p['radius_factors']:
            for query in queries:
                group=copy.deepcopy(query);group.radius*=factor;group.nsample=p['nsample']
                if not group.use_xyz:raise ValueError('KPConvX cylinder requires grouped XYZ')
                self.groups.append(group)
        width=native.in_dim if protocol=='baseline' else native.in_dim+3
        self.encoder=KernelNeighborhood(width,**options)
        self.attention=nn.Conv2d(256,1,1) if self.pooling=='attention' else None
        self.fusion=PointwiseFusion(256*len(p['radius_factors']),p['activation'])

    def _encode(self,grouped,group):
        # Native Graspness-style queries already divide aligned XYZ by radius.
        # Baseline queries use metres; normalize the local encoder inputs only.
        if not group.normalize_xyz:
            grouped=torch.cat([grouped[:,:3]/group.radius,grouped[:,3:]],dim=1)
        return self.encoder(grouped)
