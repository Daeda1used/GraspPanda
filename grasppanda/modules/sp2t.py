"""Isolated native SP2T runtime and scene-local attention integration."""
from functools import lru_cache
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def scene_attention(native, module, point):
    """Retain each scene's native window size when using PyTorch attention."""
    if module.enable_flash or len(point.offset) == 1:
        return native.original_serialized_forward(module, point)
    outputs, indices = [], []
    for scene in point.batch.unique(sorted=True):
        rows = torch.nonzero(point.batch == scene, as_tuple=False).flatten()
        order = point.serialized_inverse[:, rows].argsort(dim=1)
        local = native.Point(coord=point.coord[rows], grid_coord=point.grid_coord[rows],
            feat=point.feat[rows], offset=rows.new_tensor([len(rows)]).int(),
            serialized_order=order, serialized_inverse=order.argsort(dim=1))
        outputs.append(native.original_serialized_forward(module, local).feat)
        indices.append(rows)
    point.feat = torch.cat(outputs)[torch.argsort(torch.cat(indices))]
    return point


@lru_cache(None)
def native_module():
    from ..config import ROOT
    from ..compat import triton_driver
    from ..runtime.build_sp2t import SOURCE_COMMIT, digest, verify
    base = ROOT/'environments/native/sp2t'
    state = base/'state.json'
    if not state.is_file():
        raise ValueError('SP2T source is not prepared; run ./panda install')
    record = json.loads(state.read_text())
    directory = Path(record['directory']).resolve()
    patch = ROOT/'grasppanda/resources/patches/sp2t/native-runtime.patch'
    if (not directory.is_relative_to(base.resolve()) or record.get('source') != SOURCE_COMMIT
            or record.get('patch') != digest(patch)
            or record.get('builder') != digest(ROOT/'grasppanda/runtime/build_sp2t.py')):
        raise ValueError('SP2T runtime does not match this release; run ./panda install')
    verify(directory)
    triton_driver()
    name = 'sp2t_native'
    path = directory/name/'__init__.py'
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != path:
        raise ValueError('A different SP2T source is already loaded; restart the worker')
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        try:
            spec.loader.exec_module(package)
            native = importlib.import_module(name+'.models.sparse_proxy_point_transformer.sparse_proxy_ptv3')
        except Exception:
            for key in list(sys.modules):
                if key == name or key.startswith(name+'.'):sys.modules.pop(key, None)
            raise
    else:
        native = importlib.import_module(name+'.models.sparse_proxy_point_transformer.sparse_proxy_ptv3')
    if not hasattr(native, 'original_serialized_forward'):
        native.original_serialized_forward = native.SerializedAttention.forward
        native.SerializedAttention.forward = lambda module, point: scene_attention(native, module, point)
    if not hasattr(native, 'original_serialization'):
        native.original_serialization = native.Point.serialization
        native.Point.serialization = lambda point, order='z', depth=None, shuffle_orders=False: serialization(native, point, order, depth, shuffle_orders)
    return native


def forward_scoped(network, data):
    """Release proxy caches after each forward, including failed or nested calls."""
    initializer = native_module().ProxyInitializer
    token = initializer.GLB_CONTEXT.set({})
    try:
        return network(data)
    finally:
        initializer.GLB_CONTEXT.reset(token)


def serialization(native, point, order='z', depth=None, shuffle_orders=False):
    """Use a declared lattice depth so another scene cannot rotate Hilbert order."""
    if depth is None:depth=point.get('serialization_depth')
    return native.original_serialization(point,order,depth,shuffle_orders)


def voxelize(xyz, features, batch, grid_size, depth, grid=None):
    """Coalesce cells accurately and preserve an inverse for every original row."""
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or features.ndim!=2 or len(features)!=len(xyz) or batch.shape!=(len(xyz),):
        raise ValueError('SP2T requires aligned nonempty XYZ, features and scene IDs')
    if not xyz.is_cuda or xyz.dtype!=torch.float32 or features.dtype!=torch.float32 or features.device!=xyz.device or batch.device!=xyz.device or batch.dtype!=torch.long:
        raise ValueError('SP2T requires same-device CUDA float32 inputs and int64 scene IDs')
    if not torch.isfinite(xyz).all() or not torch.isfinite(features).all() or batch.min()<0:
        raise ValueError('SP2T requires finite inputs and nonnegative scene IDs')
    _,batch=torch.unique(batch,sorted=True,return_inverse=True)
    if (int(batch.max())+1).bit_length()+3*depth>63:raise ValueError('SP2T batch exceeds serialization capacity')
    if (xyz.detach().abs()/grid_size).max()>=2**30:raise ValueError('SP2T coordinates exceed the supported lattice range')
    if grid is None:grid=torch.floor(xyz.detach()/grid_size).long()
    elif grid.shape!=xyz.shape or grid.dtype!=torch.long or grid.device!=xyz.device or grid.abs().max()>=2**30:
        raise ValueError('SP2T grid requires aligned int64 lattice coordinates')
    else:grid=grid.clone()
    for scene in batch.unique():
        mask=batch==scene;grid[mask]-=grid[mask].amin(0)
    if grid.max()>=2**depth:raise ValueError('SP2T scene exceeds serialization_depth; increase it or grid_size')
    keys,inverse,counts=torch.unique(torch.cat([batch[:,None],grid],1),dim=0,sorted=True,return_inverse=True,return_counts=True)
    def mean(values):
        total=torch.zeros(len(keys),values.shape[1],device=values.device,dtype=torch.float64)
        return (total.index_add(0,inverse,values.double())/counts[:,None]).to(values.dtype)
    return dict(coord=mean(xyz),feat=mean(features),grid_coord=keys[:,1:],batch=keys[:,0],serialization_depth=depth),inverse


def fixed_grid_extent(module,args):
    if not module.is_first_block or (module.reuse_proxy_feat and not module.is_first):return
    coordinates,offsets=args;start=0
    for end in offsets.tolist():
        scene=coordinates[start:end];start=end
        if ((scene.amax(0)-scene.amin(0))<=0).any():
            raise ValueError('SP2T fixed_grid requires nonzero extent wherever proxy geometry is initialized; use square or fixed_size for degenerate geometry')


class SP2TFeatures(torch.nn.Module):
    def __init__(self,in_channels,out_channels,**options):
        from copy import deepcopy
        from .sp2t_options import resolve,native_config,block_layout
        super().__init__();p=self.options=resolve(options);native=native_module()
        if type(in_channels) is not int or in_channels<1 or type(out_channels) is not int or out_channels<1:
            raise ValueError('SP2T input and output widths must be positive integers')
        self.in_channels=in_channels;cfg=native_config(p)
        self.network=native.SparseProxyPTv3(in_channels=in_channels,**deepcopy(cfg))
        for module in self.network.modules():
            if isinstance(module,native.SerializedPooling):module.reduce=p['pooling']
        if any(key.startswith('block_') for key in options):
            # Preserve native stage identities and feature reuse across encoder and decoder.
            layout=block_layout(p);last_width=None;layer_index=-1
            erates=torch.linspace(0,p['drop_path'],sum(p['enc_depths'])).tolist()
            drates=torch.linspace(0,p['drop_path'],sum(p['dec_depths'])).tolist()
            for index,(side,stage,local) in enumerate(layout):
                owner=getattr(getattr(self.network,side),side+str(stage))
                active=p['proxy_start_stage']<=stage<=p['proxy_end_stage']
                if local==0 and active:
                    layer_index+=1;previous_width=last_width;last_width=p[side+'_channels'][stage]
                spa=deepcopy(cfg['spa_cfg']) if active else None
                if spa is not None:
                    spa['num_heads']=p[side+'_num_head'][stage];spa['initializer_cfg']['d_last']=previous_width
                offset=sum(p[side+'_depths'][:stage])
                rate=erates[offset+local] if side=='enc' else drates[offset+p['dec_depths'][stage]-1-local]
                def value(key,default):return p[key][index] if key in p else default
                kwargs={k:p[k] for k in ('qkv_bias','pre_norm','enable_rpe','enable_flash','upcast_attention','upcast_softmax')}
                owner._modules['block'+str(local)]=native.Block(channels=p[side+'_channels'][stage],
                    num_heads=p[side+'_num_head'][stage],patch_size=value('block_patch_sizes',p[side+'_patch_size'][stage]),
                    block_category=cfg[side+'_block_category_list'][stage],mlp_ratio=value('block_mlp_ratios',p['mlp_ratio']),
                    attn_drop=value('block_attn_drop',p['attn_drop']),proj_drop=value('block_proj_drop',p['proj_drop']),
                    drop_path=value('block_drop_path',rate),enable_checkpoint=value('block_checkpoint',p['enable_checkpoint']),
                    order_index=local%len(p['order']),cpe_indice_key=f'stage{stage}',spa_cfg=spa,
                    block_idx=local,layer_idx=layer_index,level_key=side[0]+str(stage),**kwargs)
        if p['proxy_initializer']=='fixed_grid':
            for module in self.network.modules():
                if isinstance(module,native.ProxyInitializer):module.register_forward_pre_hook(fixed_grid_extent)
        width=p['dec_channels'][0] if p['dec_channels'] else p['enc_channels'][0]
        self.projection=torch.nn.Linear(width,out_channels)

    def forward(self,xyz,features,batch,grid=None):
        if features.ndim!=2 or features.shape[1]!=self.in_channels:raise ValueError('SP2T feature width differs from its configured input')
        p=self.options;data,inverse=voxelize(xyz,features,batch,p['grid_size'],p['serialization_depth'],grid)
        if p['proxy_initializer']=='fixed_grid':
            for scene in data['batch'].unique():
                coord=data['coord'][data['batch']==scene]
                if ((coord.amax(0)-coord.amin(0))<=0).any():raise ValueError('SP2T fixed_grid requires nonzero extent on every axis; use square or fixed_size for planar inputs')
        with torch.cuda.device(xyz.device),torch.autocast('cuda',enabled=False):
            point=forward_scoped(self.network,data)
        return self.projection(point.feat)[inverse]


class SP2TBackbone(torch.nn.Module):
    def __init__(self,**options):
        super().__init__();self.features=SP2TFeatures(3,256,**options)

    def forward(self,points,end_points=None):
        from pointnet2 import _ext
        from .sampling import sample_indices
        if points.ndim!=3 or points.shape[2]!=3 or points.shape[1]<1024:raise ValueError('SP2T requires camera XYZ [B,N,3] with N >= 1024')
        points=points.contiguous();size,count,_=points.shape;xyz=points.reshape(-1,3)
        batch=torch.arange(size,device=points.device).repeat_interleave(count)
        dense=self.features(xyz,xyz,batch).reshape(size,count,256)
        indices=sample_indices(points,1024,getattr(self,'seed_sampling','upstream'),native=_ext.furthest_point_sampling,training=self.training)
        seeds=points.gather(1,indices.long()[...,None].expand(-1,-1,3)).contiguous()
        sampled=dense.gather(1,indices.long()[...,None].expand(-1,-1,256)).transpose(1,2).contiguous()
        end_points={} if end_points is None else end_points
        end_points.update(input_xyz=points,input_features=None,fp2_xyz=seeds,fp2_features=sampled,fp2_inds=indices)
        return sampled,seeds,end_points


class SparseSP2TBackbone(torch.nn.Module):
    def __init__(self,out_channels=512,voxel_size=.005,feature_channels=3,**options):
        super().__init__();self.voxel_size=voxel_size
        self.features=SP2TFeatures(3+feature_channels,out_channels,**options)

    def forward(self,sparse):
        import MinkowskiEngine as ME
        coords=sparse.C.to(device=sparse.F.device).long();xyz=coords[:,1:].float()*self.voxel_size
        grid=coords[:,1:] if self.features.options['grid_size']==self.voxel_size else None
        output=self.features(xyz,torch.cat([xyz,sparse.F],1),coords[:,0],grid)
        return ME.SparseTensor(output,coordinate_map_key=sparse.coordinate_map_key,coordinate_manager=sparse.coordinate_manager)
