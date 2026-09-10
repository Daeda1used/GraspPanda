"""Released-code PointRWKV features with explicit original-point correspondence."""
import ast
from functools import lru_cache
import hashlib
import sys
import types
import torch
from torch import nn
from .pointrwkv_options import resolve
from .pointrwkv_recurrence import parallel_recurrence

SOURCE_HASHES = {
    'models/point_rwkv.py': '82e3a7a2faec871b1d170de7579b734868d645ee04cb9b4d5712b82666943bdd',
    'models/point_rwkv_seg.py': 'e36ee02665aeb2a49429ca5bda3164ff771918209b273b64c4563b6636b15423',
    'models/pointnet2_utils.py': '55e2d794e6bfee3cb376c9ab3586a3a056c9eae81032f0d960e8759842cb25d2',
    'utils/misc.py': 'e82ab330c612470d43304940dcac334cc6e7f3e969817d508efe5a52dc8d9ceb',
}


@lru_cache(None)
def native_module():
    from ..config import ROOT
    root=ROOT/'environments/sources/cv/pointrwkv'
    for name,expected in SOURCE_HASHES.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('PointRWKV source is missing or differs from its pin; restore it with ./panda install')
    class LocalImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            return None if (node.module or '').startswith(('models.','utils.')) else node
    module=types.ModuleType('_grasppanda_pointrwkv');module.__file__=str(root/'models/point_rwkv.py')
    for name in ('utils/misc.py','models/pointnet2_utils.py','models/point_rwkv.py','models/point_rwkv_seg.py'):
        path=root/name;tree=LocalImports().visit(ast.parse(path.read_text()))
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    sys.modules[module.__name__]=module
    return module


def chunk_recurrence(self,r,k,v,w):
    return parallel_recurrence(r,k,v,w,self.u,self.grasppanda_chunk_size)


def patch_features(self,patches):
    batch,centers,neighbors,channels=patches.shape
    local=self.first_conv(patches.reshape(batch*centers,neighbors,channels).transpose(1,2))
    combined=torch.cat((local.max(2,keepdim=True).values.expand_as(local),local),1)
    return self.second_conv(combined).max(2).values.reshape(batch,centers,self.encoder_channel)


class PointRWKVFeatures(nn.Module):
    """Extra attributes are separate from camera XYZ; output follows every input row."""
    def __init__(self,feature_channels,out_channels,**options):
        super().__init__();self.options=p=resolve(options);native=native_module()
        if type(feature_channels) is not int or feature_channels<0 or type(out_channels) is not int or out_channels<1:
            raise ValueError('PointRWKV requires nonnegative attribute width and positive output width')
        self.feature_channels=feature_channels
        self.backbone=nn.Module();net=self.backbone
        net.group_modules=nn.ModuleList([native.Group(n,k) for n,k in zip(p['num_points'],p['group_sizes'])])
        net.embed_modules=nn.ModuleList()
        a,b,c=p['patch_channels']
        for width in p['stage_channels']:
            encoder=native.Encoder(width)
            if feature_channels or p['patch_channels']!=[128,256,512]:
                encoder.first_conv=nn.Sequential(nn.Conv1d(3+feature_channels,a,1),nn.BatchNorm1d(a),nn.ReLU(inplace=True),nn.Conv1d(a,b,1))
                encoder.second_conv=nn.Sequential(nn.Conv1d(2*b,c,1),nn.BatchNorm1d(c),nn.ReLU(inplace=True),nn.Conv1d(c,width,1))
                encoder.forward=types.MethodType(patch_features,encoder)
            net.embed_modules.append(encoder)
        net.pos_embed=nn.ModuleList([nn.Sequential(nn.Linear(3,128),nn.GELU(),nn.Linear(128,c)) for c in p['stage_channels']])
        rates=iter(torch.linspace(0,p['drop_path_rate'],sum(p['depths'])).tolist());net.blocks=nn.ModuleList()
        for i,width in enumerate(p['stage_channels']):
            blocks=nn.ModuleList()
            for _ in range(p['depths'][i]):
                block=native.PRWKVBlock(width,num_heads=p['stage_heads'][i],k=p['k_neighbors'][i],
                    graph_iter=p['graph_iterations'][i],ffn_ratio=p['ffn_ratios'][i],drop=p['drop'],drop_path=next(rates))
                if p['recurrence_backend']=='parallel':
                    block.spatial_mixing.grasppanda_chunk_size=p['chunk_size']
                    block.spatial_mixing._wkv_forward=types.MethodType(chunk_recurrence,block.spatial_mixing)
                blocks.append(block)
            net.blocks.append(blocks)
        net.norms=nn.ModuleList([nn.LayerNorm(c) for c in p['stage_channels']])
        net.apply(lambda module:native.PointRWKV._init_weights(net,module))
        channels=p['stage_channels'];decoder=p['decoder_channels']
        inputs=[channels[1]+channels[2],channels[0]+decoder[0],3+feature_channels+decoder[1]]
        self.propagation_layers=nn.ModuleList([native.PointNetFeaturePropagation(v,[c]*d)
            for v,c,d in zip(inputs,decoder,p['decoder_depths'])])
        self.projection=nn.Linear(decoder[-1],out_channels)

    def encode(self,scenes,attributes):
        native=native_module();net=self.backbone;features=[];centers=[]
        equal=len({len(x) for x in scenes})==1
        for i,grouper in enumerate(net.group_modules):
            if equal and not self.feature_channels:
                patches,center=grouper(torch.stack(scenes))
            else:
                patches=[];center=[]
                # Ragged scenes retain their own neighbors, without padding points.
                groups=[(torch.stack(scenes),torch.stack(attributes))] if equal else [(x[None],a[None]) for x,a in zip(scenes,attributes)]
                for xyz,attrs in groups:
                    position=native.index_points(xyz,native.fps(xyz,grouper.num_group))
                    indices=native.knn_point(grouper.group_size,xyz,position)
                    values=native.index_points(xyz,indices)-position[:,:,None]
                    if self.feature_channels:values=torch.cat((values,native.index_points(attrs,indices)),-1)
                    patches.append(values);center.append(position)
                patches=torch.cat(patches);center=torch.cat(center)
            tokens=net.embed_modules[i](patches)+net.pos_embed[i](center)
            for block in net.blocks[i]:
                if self.options['gradient_checkpointing'] and self.training:
                    from .ptv2 import checkpoint
                    tokens=checkpoint(block,center,tokens)
                else:tokens=block(center,tokens)
            features.append(net.norms[i](tokens));centers.append(center)
        return features,centers

    def decode(self,scenes,attributes,features,centers):
        layers=self.propagation_layers
        value=layers[0](centers[1],centers[2],features[1],features[2])
        value=layers[1](centers[0],centers[1],features[0],value)
        if len({len(x) for x in scenes})==1:
            xyz=torch.stack(scenes);skip=torch.cat((xyz,torch.stack(attributes)),-1)
            return layers[2](xyz,centers[0],skip,value).flatten(0,1)
        # Apply final propagation to every target, then normalize all real rows together.
        native=native_module();inputs=[]
        for i,(xyz,attrs) in enumerate(zip(scenes,attributes)):
            if centers[0].shape[1]==1:interpolated=value[i,:1].expand(len(xyz),-1)
            else:
                distances,indices=native.square_distance(xyz[None],centers[0][i:i+1]).sort(-1)
                weights=(distances[:,:,:3]+1e-8).reciprocal();weights=weights/weights.sum(-1,keepdim=True)
                neighbors=native.index_points(value[i:i+1],indices[:,:,:3])
                interpolated=(neighbors*weights[...,None]).sum(2)[0]
            inputs.append(torch.cat((xyz,attrs,interpolated),-1))
        value=torch.cat(inputs).T[None]
        for conv,bn in zip(layers[2].mlp_convs,layers[2].mlp_bns):value=torch.relu(bn(conv(value)))
        return value[0].T

    def forward(self,xyz,attributes,batch):
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or attributes.shape!=(len(xyz),self.feature_channels) or batch.shape!=(len(xyz),):
            raise ValueError('PointRWKV requires aligned nonempty XYZ, attributes and scene IDs')
        if xyz.dtype != torch.float32 or attributes.dtype!=xyz.dtype or batch.dtype!=torch.long or attributes.device!=xyz.device or batch.device!=xyz.device:
            raise ValueError('PointRWKV requires same-device float32 XYZ and attributes, with int64 scene IDs')
        if not torch.isfinite(xyz).all() or not torch.isfinite(attributes).all() or batch.min()<0:
            raise ValueError('PointRWKV requires finite inputs and nonnegative scene IDs')
        _,scene=torch.unique(batch,sorted=True,return_inverse=True);order=torch.argsort(scene,stable=True)
        counts=torch.bincount(scene).tolist()
        if min(counts)<max(self.options['group_sizes']):raise ValueError('PointRWKV patch neighborhood exceeds the input scene size')
        scenes=list(xyz[order].split(counts));attrs=list(attributes[order].split(counts))
        features,centers=self.encode(scenes,attrs)
        return self.projection(self.decode(scenes,attrs,features,centers))[torch.argsort(order)]


class PointRWKVBackbone(nn.Module):
    def __init__(self,**options):
        super().__init__();self.features=PointRWKVFeatures(0,256,**options)

    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim!=3 or points.shape[-1]!=3 or points.shape[1]<1024:
            raise ValueError('PointRWKV requires camera XYZ [B,N,3] with N >= 1024')
        points=points.contiguous();size,count,_=points.shape;xyz=points.reshape(-1,3)
        batch=torch.arange(size,device=points.device).repeat_interleave(count)
        dense=self.features(xyz,xyz.new_empty(len(xyz),0),batch).reshape(size,count,256)
        indices=sample_indices(points,1024,getattr(self,'seed_sampling','upstream'),native=_ext.furthest_point_sampling,training=self.training)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(1,indices.long()[...,None].expand(-1,-1,256)).transpose(1,2).contiguous()
        end_points={} if end_points is None else end_points
        end_points.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points


class SparsePointRWKVBackbone(nn.Module):
    def __init__(self,out_channels=512,voxel_size=.005,feature_channels=3,**options):
        super().__init__();self.voxel_size=voxel_size
        self.features=PointRWKVFeatures(feature_channels,out_channels,**options)

    def forward(self,sparse):
        import MinkowskiEngine as ME
        coords=sparse.C.to(device=sparse.F.device).long()
        xyz=coords[:,1:].to(sparse.F.dtype)*self.voxel_size
        features=self.features(xyz,sparse.F,coords[:,0])
        return ME.SparseTensor(features,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
