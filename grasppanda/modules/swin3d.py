"""Native Swin3D hierarchy with explicit grasp feature and coordinate contracts."""
from functools import lru_cache
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn
from .swin3d_options import resolve, block_stages


def native_module(record=None):
    from ..config import ROOT
    state=ROOT/'environments/native/swin3d/state.json'
    if record is None:
        if not state.is_file():raise ValueError('Swin3D is not installed; run ./panda install')
        record=json.loads(state.read_text())
    return _load(json.dumps(record,sort_keys=True),str(ROOT.resolve()))


@lru_cache(None)
def _load(serialized,root):
    from ..runtime.build_swin3d import SOURCE_COMMIT, runtime_info, digest
    record=json.loads(serialized);artifact=Path(record['root']).resolve()
    allowed=(Path(root)/'environments/native/swin3d/builds').resolve()
    if not artifact.is_relative_to(allowed) or artifact==allowed:raise ValueError('Invalid Swin3D artifact location')
    patch=Path(root)/'grasppanda/resources/patches/swin3d/native-runtime.patch'
    if record.get('source')!=SOURCE_COMMIT or record.get('runtime')!=runtime_info() or record.get('patch')!=digest(patch):
        raise ValueError('Swin3D artifact differs from the runtime or source pins; run ./panda install')
    hashes=record.get('hashes',{})
    actual={str(p.relative_to(artifact)) for p in (artifact/'python').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    required={'python/Swin3D/models/Swin3D.py','python/Swin3D/modules/swin3d_layers.py','python/Swin3D/sparse_dl/knn/knn.py'}
    if not required<=actual or not actual<=hashes.keys() or not any(n.endswith('.so') and '/attn_cuda.' in n for n in actual) or not any(n.endswith('.so') and '/knn_cuda.' in n for n in actual):
        raise ValueError('Swin3D artifact is incomplete; run ./panda install')
    for name,expected in hashes.items():
        path=(artifact/name).resolve()
        if not path.is_relative_to(artifact) or not path.is_file() or digest(path)!=expected:raise ValueError('Swin3D artifact is modified; run ./panda install')
    for name,module in list(sys.modules.items()):
        if name=='Swin3D' or name.startswith('Swin3D.'):
            if not Path(module.__file__).resolve().is_relative_to(artifact/'python'):
                raise ValueError('A different Swin3D artifact is loaded; start a fresh worker')
    directory=str(artifact/'python');sys.path.insert(0,directory)
    try:
        native=importlib.import_module('Swin3D.modules.swin3d_layers')
        native.Swin3DUNet=importlib.import_module('Swin3D.models.Swin3D').Swin3DUNet
        return native
    except Exception:
        sys.path.remove(directory)
        for name in list(sys.modules):
            if name=='Swin3D' or name.startswith('Swin3D.'):sys.modules.pop(name,None)
        raise


def _initialize(module):
    from timm.layers import trunc_normal_
    if isinstance(module,nn.Linear):
        trunc_normal_(module.weight,std=.02)
        if module.bias is not None:nn.init.zeros_(module.bias)
    elif isinstance(module,(nn.LayerNorm,nn.BatchNorm1d)):
        nn.init.ones_(module.weight);nn.init.zeros_(module.bias)


def _checkpoint_block(self,features,arguments):
    if self.training and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.grasppanda_forward,features,arguments,use_reentrant=False)
    return self.grasppanda_forward(features,arguments)


class Swin3DFeatures(nn.Module):
    def __init__(self,feature_channels,out_channels,_native=None,**options):
        super().__init__();p=self.options=resolve(options);native=_native or native_module()
        if type(feature_channels) is not int or feature_channels<0 or type(out_channels) is not int or out_channels<1:
            raise ValueError('Swin3D requires nonnegative attribute width and positive output width')
        self.feature_channels=feature_channels
        levels=len(p['channels']);crse='XYZ_NORM' if p['rpe_features']=='xyz_normals' else 'XYZ'
        net=self.network=native.Swin3DUNet(p['depths'],p['channels'],p['heads'],p['window_sizes'],p['quant_sizes'][0],
            drop_path_rate=p['drop_path_rate'],up_k=3,num_layers=levels,num_classes=out_channels,
            stem_transformer=p['stem_transformer'],first_down_stride=p['strides'][0] if levels>1 else 2,
            upsample='linear_attn',knn_down=True,in_channels=3+feature_channels,cRSE=crse,fp16_mode=0)
        layers={i:layer for i,layer in zip(range(net.layer_start,levels),net.layers)}
        for stage in range(levels-1):
            owner=net if stage==0 and not p['stem_transformer'] else layers[stage]
            current=owner.downsample;stride=p['strides'][stage]
            kind=native.GridKNNDownsample if p['downsample'][stage]=='knn' else native.GridDownsample
            if not isinstance(current,kind) or current.stride!=stride:
                owner.downsample=kind(p['channels'][stage],p['channels'][stage+1],kernel_size=stride,stride=stride)
                owner.downsample.apply(_initialize)
            if p['downsample'][stage]=='knn':
                owner.downsample.k=p['knn_neighbors'][stage]
                owner.downsample.pool=nn.MaxPool1d(p['knn_neighbors'][stage])
        decoder={}
        for stage,up in zip(reversed(range(levels-1)),net.upsamples):
            up.up_k=p['up_neighbors'][stage];depth=p['decoder_depths'][stage]
            if depth==0:
                up.attn=False;del up.block
            else:
                if depth!=1:
                    up.block=native.BasicLayer(dim=p['channels'][stage],depth=depth,num_heads=p['heads'][stage],
                        window_size=p['window_sizes'][stage],quant_size=p['quant_sizes'][stage],
                        drop_path=p['decoder_drop_path'],cRSE=crse,fp16_mode=0)
                    up.block.apply(_initialize)
                decoder[stage]=up.block
        # Flat overrides follow encoder stages, then the decoder's coarse-to-fine execution.
        all_layers=[(i,layers[i]) for i in sorted(layers)]+[(i,decoder[i]) for i in sorted(decoder,reverse=True)]
        stages=block_stages(p)
        rates=torch.linspace(0,p['drop_path_rate'],sum(p['depths'])).tolist()+[p['decoder_drop_path']]*sum(p['decoder_depths'])
        index=0
        for stage,layer in all_layers:
            layer.quant_size=p['quant_sizes'][stage]
            for local,block in enumerate(layer.blocks):
                heads=p.get('block_heads',[p['heads'][i] for i in stages])[index]
                ratio=p.get('block_mlp_ratios',[p['mlp_ratios'][i] for i in stages])[index]
                rate=p.get('block_drop_path',rates)[index]
                if (heads!=block.attn.num_heads or ratio!=4 or not p['qkv_bias'][stage] or block.attn.quant_size!=p['quant_sizes'][stage]):
                    block=native.SwinTransformerBlock(p['channels'][stage],heads,p['window_sizes'][stage],p['quant_sizes'][stage],
                        drop_path=rate,mlp_ratio=ratio,qkv_bias=p['qkv_bias'][stage],cRSE=crse,fp16_mode=0)
                    block.apply(_initialize);layer.blocks[local]=block
                block.drop_path=native.DropPath(rate) if rate else nn.Identity()
                block.attn.proj_drop.p=p.get('block_projection_dropout',[p['projection_dropout'][i] for i in stages])[index]
                block.mlp.drop.p=p.get('block_mlp_dropout',[p['mlp_dropout'][i] for i in stages])[index]
                block.mlp.act={'gelu':nn.GELU,'relu':nn.ReLU,'silu':nn.SiLU}[p['activation']]()
                if p['gradient_checkpointing']:
                    block.grasppanda_forward=block.forward
                    block.forward=types.MethodType(_checkpoint_block,block)
                index+=1
        for module in net.modules():
            if isinstance(module,nn.BatchNorm1d):module.eps=p['bn_eps'];module.momentum=p['bn_momentum']
            if isinstance(module,nn.LayerNorm):module.eps=p['norm_eps']

    def forward(self,xyz,attributes,batch,normals=None,grid=None):
        import MinkowskiEngine as ME
        if xyz.ndim!=2 or xyz.shape[1]!=3 or attributes.shape!=(len(xyz),self.feature_channels) or batch.shape!=(len(xyz),) or not len(xyz):
            raise ValueError('Swin3D requires aligned nonempty XYZ, attributes and scene IDs')
        if not xyz.is_cuda or xyz.dtype!=torch.float32 or attributes.dtype!=torch.float32 or attributes.device!=xyz.device or batch.device!=xyz.device or batch.dtype!=torch.long:
            raise ValueError('Swin3D requires same-device CUDA float32 XYZ/attributes and int64 scene IDs')
        if not torch.isfinite(xyz).all() or not torch.isfinite(attributes).all() or batch.min()<0:
            raise ValueError('Swin3D requires finite inputs and nonnegative scene IDs')
        _,scene=torch.unique(batch,sorted=True,return_inverse=True)
        scale=self.options['grid_size']
        if (xyz.detach().abs()/scale).max()>=2**30:raise ValueError('Swin3D coordinates exceed the supported lattice range')
        if grid is None:cells=torch.floor(xyz.detach()/scale).long()
        else:
            if grid.shape!=xyz.shape or grid.dtype!=torch.long or grid.device!=xyz.device or grid.abs().max()>=2**30:
                raise ValueError('Swin3D grid requires aligned int64 lattice coordinates')
            cells=grid
        keys,inverse,counts=torch.unique(torch.cat([scene[:,None],cells],1),dim=0,sorted=True,return_inverse=True,return_counts=True)
        def mean(values):
            # Accurate voxel means keep duplicated rows from moving discrete RPE bins.
            accumulated=torch.zeros(len(keys),values.shape[1],device=values.device,dtype=torch.float64)
            return (accumulated.index_add(0,inverse,values.double())/counts[:,None]).to(values.dtype)
        features=mean(torch.cat([xyz,attributes],1))
        position=keys[:,1:].float() if grid is not None else mean(xyz.detach())/scale
        position=torch.cat([keys[:,:1].float(),position],1)
        if self.options['rpe_features']=='xyz_normals':
            if normals is None or normals.shape!=xyz.shape or normals.dtype!=torch.float32 or normals.device!=xyz.device or not torch.isfinite(normals).all() or normals.abs().max()>1.00001:
                raise ValueError('Swin3D normal RPE requires real aligned normals in [-1,1]')
            # The author layout reserves columns 4:7 for RGB; XYZ_NORM never reads them.
            position=torch.cat([position,position.new_zeros(len(keys),3),mean(normals.detach()).clamp(-1,1)],1)
        with torch.cuda.device(xyz.device):
            sparse=ME.SparseTensor(features,coordinates=keys.int(),device=xyz.device)
            aligned=position[sparse.unique_index]
            if not torch.equal(sparse.C,keys[sparse.unique_index].int()):raise ValueError('Swin3D sparse coordinates lost their feature correspondence')
            coordinates=ME.SparseTensor(aligned,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
            # Native kernels use float32 features in the shared grasp runtime.
            with torch.autocast('cuda',enabled=False):output=self.network(sparse,coordinates)
        return output[sparse.inverse_mapping][inverse]


class Swin3DBackbone(nn.Module):
    def __init__(self,**options):
        super().__init__();self.features=Swin3DFeatures(0,256,**options)
        if self.features.options['rpe_features']!='xyz':raise ValueError('Dense grasp backbones do not supply normals for Swin3D RPE')

    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim!=3 or points.shape[2]!=3 or points.shape[1]<1024:raise ValueError('Swin3D requires camera XYZ [B,N,3] with N >= 1024')
        points=points.contiguous();size,count,_=points.shape;xyz=points.reshape(-1,3)
        batch=torch.arange(size,device=points.device).repeat_interleave(count)
        dense=self.features(xyz,xyz.new_empty(len(xyz),0),batch).reshape(size,count,256)
        indices=sample_indices(points,1024,getattr(self,'seed_sampling','upstream'),native=_ext.furthest_point_sampling,training=self.training)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(1,indices.long()[...,None].expand(-1,-1,256)).transpose(1,2).contiguous()
        end_points={} if end_points is None else end_points
        end_points.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points


class SparseSwin3DBackbone(nn.Module):
    def __init__(self,out_channels=512,voxel_size=.005,feature_channels=3,normal_features=False,**options):
        super().__init__();self.voxel_size=voxel_size;self.normal_features=normal_features
        self.features=Swin3DFeatures(feature_channels,out_channels,**options)
        if self.features.options['rpe_features']=='xyz_normals' and (not normal_features or feature_channels!=6):
            raise ValueError('Normal-aware Swin3D RPE requires FineGrasp with native normals enabled')

    def forward(self,sparse):
        import MinkowskiEngine as ME
        coords=sparse.C.to(device=sparse.F.device).long();xyz=coords[:,1:].float()*self.voxel_size
        grid=coords[:,1:] if self.features.options['grid_size']==self.voxel_size else None
        normals=sparse.F[:,3:6] if self.normal_features else None
        output=self.features(xyz,sparse.F,coords[:,0],normals=normals,grid=grid)
        return ME.SparseTensor(output,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
