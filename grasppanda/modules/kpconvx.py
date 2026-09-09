"""Official KPConvX feature hierarchy with native grasp point correspondence."""
import ast
import fcntl
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import torch
from torch import nn
from .kpconvx_options import resolve

SOURCE_HASHES = {
    'kpconvx_base.py': '9cb156f839c3890018fe24a52ce78327926f397037452277f4b63e673032b26a',
    'utils/generic_blocks.py': '60782b4663585db16fd1ce9c7dc8e14f41c5fa76cfd8d3647cd8f16f0c6489a0',
    'utils/kpconv_blocks.py': '784a839d97eac7af49eac044847ce1c720ff5bfaacb8c18152b6e4639bfb3a94',
    'utils/kpnext_blocks.py': '7b4e761a0a0cfc0143f111ab7d0ed5889070531c5e510e8739ca133783125189',
    'utils/kernel_points.py': 'af768d26a9bbe510b67e3e1f90d3e1007b2150fd2a4fec334db643ac23c0fd73',
    'utils/ply.py': '8598cd1d380a0a1c0ce6c844894a08e441ef7cadecdec8bd82b91f3b531025f1',
    'utils/torch_pyramid.py': '9c968e4d79ae1e3328ec57c8536c8354e2ca06be598c14e0aca99ec4176a024d',
    'utils/dispositions/k_043_center_3D_0.ply': '7f153d5bb9aef88a5954463e36afb75b2acb564a9b0eb84556d9f574251d59ee',
}


def kernel_directory(shell_sizes, dimension, fixed):
    """Generate each exact shell layout once without consuming the experiment RNG."""
    from ..config import ROOT
    if dimension != 3 or fixed != 'center' or len(shell_sizes)<2 or shell_sizes[0]!=1:
        raise ValueError('KPConvX requires centered 3D kernels with at least two shells')
    directory=ROOT/'environments/cache/kpconvx'/('-'.join(map(str,shell_sizes)))
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'k_{sum(shell_sizes):03d}_center_3D_0.ply'
    native=sys.modules['_grasppanda_kpconvx.utils.kernel_points']
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.is_file():
            temporary=directory/'kernel.tmp.ply'
            if tuple(shell_sizes)==(1,14,28):
                source=Path(native.__file__).parent/'dispositions'/target.name
                shutil.copyfile(source,temporary)
            else:
                # The author generator initializes on the CPU, then optimizes on CUDA.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(0)
                    candidates,gradients=native.shell_kernel_generator(1.,shell_sizes,num_kernels=100,
                        dimension=3,verbose=0)
                points=candidates[np.argmin(gradients[-1])]
                native.write_ply(str(temporary),(points,np.zeros(len(points))),['x','y','z','v'])
            temporary.replace(target)
        data=native.read_ply(str(target))
        coords=np.column_stack([data[k] for k in ('x','y','z')])
        if len(coords)!=sum(shell_sizes) or not np.isfinite(coords).all():
            raise ValueError('Invalid KPConvX kernel cache; remove its directory and retry')
    return str(directory)


def modulation_norm(module, values):
    if getattr(module,'modulation_scope','native')=='native':
        return module.grpnorm(values.T[None]).squeeze(0).T
    return module.grpnorm(values)


@lru_cache(None)
def native_module():
    from ..config import ROOT
    try:
        import _grasppanda_pointops as pointops
    except (ImportError, OSError) as error:
        raise ValueError('KPConvX point operators are missing or incompatible; run ./panda install') from error
    root=ROOT/'environments/sources/cv/kpconvx/Pointcept-wrapper/models/kpconvx'
    for name,expected in SOURCE_HASHES.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('KPConvX source is missing or differs from the pinned revision; run ./panda install')
    package='_grasppanda_kpconvx'
    for suffix in ('','utils'):
        name=package+('.'+suffix if suffix else '')
        module=types.ModuleType(name); module.__path__=[str(root/suffix)]
        sys.modules[name]=module
    loaded=[]
    try:
        for relative in ('utils/ply.py','utils/kernel_points.py','utils/generic_blocks.py',
                         'utils/kpconv_blocks.py','utils/kpnext_blocks.py','utils/torch_pyramid.py','kpconvx_base.py'):
            path=root/relative; name=package+'.'+relative[:-3].replace('/','.')
            module=types.ModuleType(name); module.__file__=str(path); tree=ast.parse(path.read_text())
            body=[]
            for node in tree.body:
                if isinstance(node,ast.ImportFrom):
                    source=node.module or ''
                    if source=='symbol' or source=='pointcept.models.builder': continue
                    if source.endswith(('gpu_subsampling','gpu_neigbors','cpp_funcs')): continue
                    if source=='pointcept.models.utils': continue
                    node.module=source.replace('pointcept.models.kpconvx',package)
                if isinstance(node,ast.Import) and any(a.name=='pointops' for a in node.names): continue
                if isinstance(node,ast.ClassDef): node.decorator_list=[]
                body.append(node)
            tree.body=body
            if relative=='utils/kernel_points.py':
                function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_kernels')
                function.body[0].value=ast.parse('kernel_directory(shell_sizes, dimension, fixed)',mode='eval').body
                module.kernel_directory=kernel_directory
                # The released utility calls init_gpu without defining/importing it.
                module.init_gpu=lambda: torch.device('cuda',torch.cuda.current_device())
            elif relative=='utils/kpnext_blocks.py':
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KPConvX')
                forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
                branch=next(n for n in forward.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self.mod_grp_norm')
                branch.body=ast.parse('modulations = modulation_norm(self, modulations)').body
                module.modulation_norm=modulation_norm
            elif relative=='utils/torch_pyramid.py':
                module.pointops=pointops
                module.offset2batch=lambda offset: torch.arange(len(offset),device=offset.device).repeat_interleave(torch.diff(offset,prepend=offset.new_zeros(1)))
                module.batch2offset=lambda batch: batch.bincount().cumsum(0)
                # This adapter always uses the author's GPU grid-pooling branch.
                module.radius_search_pack_mode=None
            elif relative=='kpconvx_base.py':
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KPConvXBase')
                constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
                constructor.args.args.append(ast.arg(arg='stage_channels'))
                constructor.args.defaults.append(ast.Constant(value=None))
                at=next(i for i,n in enumerate(constructor.body) if isinstance(n,ast.For)
                    and any(isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name)
                    and x.value.id=='layer_C' for x in ast.walk(n)))+1
                constructor.body[at:at]=ast.parse('if stage_channels is not None: layer_C = list(stage_channels)').body
                # A single-stage hierarchy needs no next-stage width or pooling.
                for node in ast.walk(tree):
                    if isinstance(node,ast.IfExp) and ast.unparse(node.body)=='layer_C[1]':
                        node.test=ast.BoolOp(op=ast.And(),values=[ast.parse('self.num_layers > 1',mode='eval').body,node.test])
            sys.modules[name]=module;loaded.append(name)
            exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
        return module
    except Exception:
        for name in loaded+[package,package+'.utils']: sys.modules.pop(name,None)
        raise


class KPConvXFeatures(nn.Module):
    def __init__(self,in_channels,out_channels,**options):
        super().__init__()
        from ..module_options import validate_options
        validate_options('graspnet_baseline','backbone','kpconvx',options)
        p=resolve(options); activation=p.pop('activation');scope=p.pop('modulation_scope')
        self.network=native_module().KPConvXBase(input_channels=in_channels,num_classes=out_channels,**p)
        for module in list(self.network.modules()):
            if hasattr(module,'mod_grp_norm'): module.modulation_scope=scope
            if activation!='leaky_relu':
                for key,child in list(module.named_children()):
                    if isinstance(child,nn.LeakyReLU):
                        setattr(module,key,{'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}[activation]())

    def forward(self,xyz,features,batch):
        if xyz.ndim!=2 or xyz.shape[1]!=3 or features.ndim!=2 or len(features)!=len(xyz) or batch.shape!=(len(xyz),):
            raise ValueError('KPConvX expects aligned XYZ [N,3], features [N,C] and batch IDs [N]')
        if xyz.dtype!=torch.float32 or features.dtype!=torch.float32 or not xyz.is_cuda or features.device!=xyz.device or batch.dtype!=torch.long or batch.device!=xyz.device:
            raise ValueError('KPConvX requires CUDA float32 coordinates/features and int64 batch IDs on the same device')
        if not torch.isfinite(xyz).all() or not torch.isfinite(features).all():
            raise ValueError('KPConvX requires finite coordinates and features')
        if not len(batch) or batch.min()<0 or batch.max()>=len(batch) or (batch.bincount()==0).any():
            raise ValueError('KPConvX requires nonempty scenes with consecutive batch IDs')
        order=torch.argsort(batch,stable=True)
        data=dict(coord=xyz[order].contiguous(),feat=features[order].contiguous(),offset=batch.bincount().cumsum(0).int())
        with torch.cuda.device(xyz.device): dense=self.network(data)
        return dense.new_empty(dense.shape).index_copy(0,order,dense)


class KPConvXBackbone(nn.Module):
    def __init__(self,**options):
        super().__init__();self.features=KPConvXFeatures(3,256,**options)

    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        if points.ndim!=3 or points.shape[2]!=3 or points.shape[1]<1024:
            raise ValueError('KPConvX requires camera XYZ [B,N,3] with N >= 1024')
        points=points.contiguous();size,count,_=points.shape
        batch=torch.arange(size,device=points.device).repeat_interleave(count);xyz=points.reshape(-1,3)
        dense=self.features(xyz,xyz,batch).reshape(size,count,256)
        indices=_ext.furthest_point_sampling(points,1024)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(1,indices.long()[...,None].expand(-1,-1,256)).transpose(1,2).contiguous()
        end_points={} if end_points is None else end_points
        end_points.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points


class SparseKPConvXBackbone(nn.Module):
    def __init__(self,out_channels=512,voxel_size=.005,feature_channels=3,**options):
        super().__init__();self.voxel_size=voxel_size
        self.features=KPConvXFeatures(3+feature_channels,out_channels,**options)

    def forward(self,sparse):
        import MinkowskiEngine as ME
        coords=sparse.C.to(device=sparse.F.device).long();xyz=coords[:,1:].to(sparse.F.dtype)*self.voxel_size
        features=self.features(xyz,torch.cat([xyz,sparse.F],1),coords[:,0])
        return ME.SparseTensor(features,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
