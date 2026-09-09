"""Native PointCNN++ encoders with configurable residual blocks and point ordering."""
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from .pointcnnpp_options import resolve


def native_module(record=None):
    from ..config import ROOT
    state=ROOT/'environments/native/pointcnnpp/state.json'
    if record is None:
        if not state.is_file():raise ValueError('PointCNN++ is not installed; run ./panda install')
        record=json.loads(state.read_text())
    return _load(json.dumps(record,sort_keys=True),str(ROOT.resolve()))


@lru_cache(None)
def _load(serialized,root):
    from ..compat import triton_driver
    from ..runtime.build_pointcnnpp import SOURCE_COMMIT, CUTLASS_COMMIT, runtime_info
    record=json.loads(serialized); artifact=Path(record['root']).resolve()
    allowed=(Path(root)/'environments/native/pointcnnpp/builds').resolve()
    if not artifact.is_relative_to(allowed) or artifact==allowed:
        raise ValueError('PointCNN++ artifact is outside its installation directory')
    if record.get('runtime')!=runtime_info() or record.get('source')!=SOURCE_COMMIT or record.get('cutlass')!=CUTLASS_COMMIT:
        raise ValueError('PointCNN++ artifact does not match the shared runtime or source pins; run ./panda install')
    hashes=record.get('hashes',{})
    if not hashes or 'python/_grasppanda_pointcnnpp/native_unet.py' not in hashes:
        raise ValueError('PointCNN++ artifact has no complete source manifest')
    actual={str(p.relative_to(artifact)) for p in (artifact/'python').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    if not actual or not actual.issubset(hashes):raise ValueError('PointCNN++ artifact contains unregistered runtime files')
    if not any(name.endswith('.so') for name in actual):raise ValueError('PointCNN++ native operators are missing')
    for name,expected in hashes.items():
        path=(artifact/name).resolve()
        if not path.is_relative_to(artifact) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('PointCNN++ artifact is incomplete or modified; run ./panda install')
    python=artifact/'python'
    for name in ('sparse_engines_cuda','_grasppanda_pointcnnpp'):
        if name in sys.modules:
            path=Path(sys.modules[name].__file__).resolve()
            if not path.is_relative_to(python):raise ValueError('A different PointCNN++ artifact is already loaded; start a fresh worker')
    triton_driver();sys.path.insert(0,str(python))
    try:return importlib.import_module('_grasppanda_pointcnnpp.native_unet')
    except Exception:
        sys.path.remove(str(python))
        for name in list(sys.modules):
            if name=='_grasppanda_pointcnnpp' or name.startswith('_grasppanda_pointcnnpp.'):sys.modules.pop(name,None)
        raise


def _activation(name):
    return {'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}[name]()


def _before_block(module,args):
    _,metadata=args
    signature=(module.conv1.kernel_size_3,module.receptive_field_scaler)
    if getattr(metadata,'_grasppanda_geometry',None)!=signature:metadata.dirty_triplets()


def _after_block(module,args,result):
    result[1]._grasppanda_geometry=(module.conv2.kernel_size_3,module.receptive_field_scaler)


def _after_upsample(module,args,result):
    result[1]._grasppanda_geometry=None


class PointCNNFeatures(nn.Module):
    def __init__(self,in_channels,out_channels,_native=None,**options):
        super().__init__();p=resolve(options);native=_native or native_module()
        self.in_channels=in_channels
        self.network=native.ResUNetPointCNNpp(in_channels=in_channels,num_classes=out_channels,
            voxel_size=p['grid_size'],base_channels=p['base_channels'],channels=p['channels'],
            layers=p['depths'],normalize_feature=p['normalize_features'])
        self.network.relu=_activation(p['activation'])
        blocks=[block for name in ('block1','enc_block1','enc_block2','enc_block3',
            'block4_tr','block3_tr','block2_tr','block1_tr') for block in getattr(self.network,name)]
        for block,kernel,scaler,activation in zip(blocks,p['block_kernel_sizes'],p['block_radius_scalers'],p['block_activations']):
            block.receptive_field_scaler=scaler;block.relu=_activation(activation)
            if kernel!=3:
                for name in ('conv1','conv2'):
                    old=getattr(block,name)
                    conv=native.PointConv3d(old.in_channels,old.out_channels,kernel_size=kernel,bias=False)
                    nn.init.trunc_normal_(conv.weight,std=.02)
                    setattr(block,name,conv)
            block.register_forward_pre_hook(_before_block)
            block.register_forward_hook(_after_block)
        for module in self.network.modules():
            if isinstance(module,nn.BatchNorm1d):module.eps=p['bn_eps'];module.momentum=p['bn_momentum']
            elif isinstance(module,native.Upsample):module.register_forward_hook(_after_upsample)

    def forward(self,xyz,features,batch,grid=None):
        if (xyz.ndim!=2 or xyz.shape[1]!=3 or features.shape!=(len(xyz),self.in_channels)
            or batch.shape!=(len(xyz),) or not len(xyz)):
            raise ValueError('PointCNN++ requires aligned coordinates, features and scene IDs')
        if (not xyz.is_cuda or xyz.dtype!=torch.float32 or features.dtype!=torch.float32
            or features.device!=xyz.device or batch.device!=xyz.device or batch.dtype!=torch.long):
            raise ValueError('PointCNN++ requires CUDA float32 coordinates/features and int64 scene IDs')
        if not torch.isfinite(xyz).all() or not torch.isfinite(features).all() or batch.min()<0:
            raise ValueError('PointCNN++ requires finite inputs and nonnegative scene IDs')
        _,scene=torch.unique(batch,sorted=True,return_inverse=True)
        order=torch.argsort(scene,stable=True);counts=torch.bincount(scene)
        with torch.cuda.device(xyz.device):
            output=self.network({'coord':xyz[order].contiguous(),'feat':features[order].contiguous(),
                'offset':counts.cumsum(0)})
        if len(output)!=len(xyz):raise ValueError('PointCNN++ did not restore every input point')
        return output[torch.argsort(order)]


class PointCNNBackbone(nn.Module):
    def __init__(self,**options):
        super().__init__();self.features=PointCNNFeatures(3,256,**options)

    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim!=3 or points.shape[2]!=3 or points.shape[1]<1024:
            raise ValueError('PointCNN++ requires camera XYZ [B,N,3] with N >= 1024')
        points=points.contiguous();size,count,_=points.shape;xyz=points.reshape(-1,3)
        batch=torch.arange(size,device=points.device).repeat_interleave(count)
        dense=self.features(xyz,xyz,batch).reshape(size,count,256)
        indices=sample_indices(points,1024,getattr(self,'seed_sampling','upstream'),
            native=_ext.furthest_point_sampling,training=self.training)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(1,indices.long()[...,None].expand(-1,-1,256)).transpose(1,2).contiguous()
        end_points={} if end_points is None else end_points
        end_points.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points


class SparsePointCNNBackbone(nn.Module):
    def __init__(self,out_channels=512,voxel_size=.005,feature_channels=3,**options):
        super().__init__();self.voxel_size=voxel_size
        self.features=PointCNNFeatures(3+feature_channels,out_channels,**options)

    def forward(self,sparse):
        import MinkowskiEngine as ME
        coords=sparse.C.to(device=sparse.F.device).long();xyz=coords[:,1:].to(sparse.F.dtype)*self.voxel_size
        features=self.features(xyz,torch.cat([xyz,sparse.F],1),coords[:,0])
        return ME.SparseTensor(features,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
